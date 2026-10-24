"""Localhost-only HTTP backend for the LegumePhC web Studio.

Projects are single-file ``.lphc`` documents (see
:mod:`legumephc.studio.document`).  Each open document is unpacked into a
private workspace; studies compute into that workspace without saving first
(as in COMSOL) and *Save* packs it back into the one file.

Every route delegates to the toolkit-independent Studio layer.  Calculations
always run in the exact worker child process; nothing here re-implements a
solver.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..studio import forms
from ..studio.document import (
    LPHC_SUFFIX, DocumentError, attach_solution, clear_solution, convert_legacy_project, new_workspace,
    normalize_document_project, open_document, read_document_meta, save_document, write_workspace_project,
)
from ..studio.inspection import record_view
from ..studio.plotting import export_figure, plot_record, sample_cell_polygons
from ..studio.preview import preview_geometry
from ..studio.project import (
    PRESET_SUFFIX, PROJECT_SUFFIX, add_calculation, apply_preset, copy_calculation, delete_calculation, load_preset,
    new_preset, new_project, project_records_dir, rename_calculation, save_preset,
)
from ..studio.worker import build_worker_request
from ..units import frequency_factor
from .jsonify import real_arrays, to_json
from .runs import RunRegistry


ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
PREVIEW_VIEWS = ("unit_cell", "motif_array", "lattice_sites", "reciprocal_bz", "epsilon")
MAX_VIEW_ARRAY = 20000


@dataclass
class OpenDocument:
    id: str
    workspace: Path
    path: Path | None


class ProjectBody(BaseModel):
    project: dict[str, Any]


class OpenBody(BaseModel):
    path: str


class SaveBody(BaseModel):
    project: dict[str, Any]
    path: str | None = None


class AttachBody(BaseModel):
    project: dict[str, Any]
    entry: dict[str, Any]


class ClearBody(BaseModel):
    project: dict[str, Any]
    calculation_id: str | None = None


class ModelFormBody(BaseModel):
    model: dict[str, Any]
    form: dict[str, Any] = Field(default_factory=dict)
    ui_state: dict[str, Any] = Field(default_factory=dict)
    parameters: list[dict[str, Any]] = Field(default_factory=list)


class SitesBody(BaseModel):
    model: dict[str, Any]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    representation: str = "n"
    parameters: list[dict[str, Any]] = Field(default_factory=list)


class PreviewBody(BaseModel):
    model: dict[str, Any]
    views: list[str] = Field(default_factory=lambda: list(PREVIEW_VIEWS))
    size: int = 96


class CalculationBody(BaseModel):
    project: dict[str, Any]
    calculation_id: str | None = None
    name: str | None = None
    operation: str = "frequency_at_k"


class RecordBody(BaseModel):
    record: str
    style: dict[str, Any] = Field(default_factory=dict)
    fmt: str = "png"


class PresetBody(BaseModel):
    name: str
    model: dict[str, Any]
    calculation: dict[str, Any]
    ui_state: dict[str, Any] = Field(default_factory=dict)


def create_app(projects_dir: str | Path | None = None, *, worker_cwd: str | Path | None = None, workspace_root: str | Path | None = None) -> FastAPI:
    projects = Path(projects_dir or ROOT / "projects").resolve()
    presets = projects / "presets"
    workspaces = Path(workspace_root or Path(tempfile.gettempdir()) / "legumephc-studio" / uuid.uuid4().hex[:8]).resolve()
    registry = RunRegistry(Path(worker_cwd or ROOT).resolve())
    documents: dict[str, OpenDocument] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for run in list(registry.runs.values()):
            if run.state == "running":
                registry.cancel(run.id)
        shutil.rmtree(workspaces, ignore_errors=True)

    app = FastAPI(title="LegumePhC Studio", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.projects_dir = projects
    app.state.runs = registry
    app.state.documents = documents

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        payload: dict[str, Any] = {"detail": str(exc)}
        if isinstance(exc, (forms.SiteError, forms.ParameterError)):
            payload.update({"index": exc.index, "field": exc.field})
        if isinstance(exc, forms.SiteError):
            payload["site_index"] = exc.index
        return JSONResponse(status_code=422, content=payload)

    def inside_projects(path: str, suffix: str) -> Path:
        candidate = Path(path)
        target = (candidate if candidate.is_absolute() else projects / candidate).resolve()
        try:
            target.relative_to(projects)
        except ValueError as exc:
            raise HTTPException(403, "documents must live inside the Studio projects folder") from exc
        if not target.name.endswith(suffix):
            target = target.with_name(target.name + suffix)
        return target

    def relative(path: Path) -> str:
        return str(path.relative_to(projects)).replace("\\", "/")

    def document(doc_id: str) -> OpenDocument:
        found = documents.get(doc_id)
        if found is None:
            raise HTTPException(404, "document is not open (the server may have restarted); reopen it")
        return found

    def register(workspace: Path, path: Path | None) -> OpenDocument:
        doc = OpenDocument(uuid.uuid4().hex[:12], workspace, path)
        documents[doc.id] = doc
        return doc

    def opened(doc: OpenDocument, project: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {"doc_id": doc.id, "path": relative(doc.path) if doc.path else None, "project": dict(project), **extra}

    def variables(parameters: list[dict[str, Any]]) -> dict[str, float]:
        return forms.evaluate_parameters(parameters)

    # ------------------------------------------------------------------ meta
    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "operations": forms.OPERATIONS,
            "lattice_summary": forms.LATTICE_SUMMARY,
            "preview_views": PREVIEW_VIEWS,
            "frequency_units": ["Normalized", "GHz", "THz"],
            "length_units": ["nm", "μm", "m"],
            "shapes": ["Circle", "Triangle", "Square", "Regular polygon"],
            "projects_dir": str(projects),
            "document_suffix": LPHC_SUFFIX,
        }

    # ------------------------------------------------------------- documents
    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        if not projects.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in projects.rglob(f"*{LPHC_SUFFIX}"):
            try:
                info = read_document_meta(path)
            except Exception:  # unreadable file still listed, opening reports why
                info = {}
            items.append({"path": relative(path), "name": path.name[: -len(LPHC_SUFFIX)], "kind": "document", "modified": path.stat().st_mtime,
                          "size": path.stat().st_size, "saved_at": info.get("saved_at"), "studies": info.get("studies"), "results": info.get("record_count")})
        for path in projects.glob(f"*{PROJECT_SUFFIX}"):
            converted = path.with_name(path.name[: -len(PROJECT_SUFFIX)] + LPHC_SUFFIX)
            items.append({"path": relative(path), "name": path.name[: -len(PROJECT_SUFFIX)], "kind": "legacy", "modified": path.stat().st_mtime,
                          "size": path.stat().st_size, "converted": converted.exists()})
        return sorted(items, key=lambda item: item["modified"], reverse=True)

    @app.post("/api/documents/new")
    def new_document(name: str = "Untitled") -> dict[str, Any]:
        doc = register(new_workspace(workspaces), None)
        project = normalize_document_project(new_project(name))
        write_workspace_project(doc.workspace, project)
        return opened(doc, project)

    @app.post("/api/documents/open")
    def open_existing(body: OpenBody) -> dict[str, Any]:
        converted_from = None
        if body.path.endswith(PROJECT_SUFFIX):
            legacy = inside_projects(body.path, PROJECT_SUFFIX)
            if not legacy.is_file():
                raise HTTPException(404, f"project not found: {body.path}")
            target = legacy.with_name(legacy.name[: -len(PROJECT_SUFFIX)] + LPHC_SUFFIX)
            if not target.exists():
                convert_legacy_project(legacy, target)
                converted_from = relative(legacy)
        else:
            target = inside_projects(body.path, LPHC_SUFFIX)
        if not target.is_file():
            raise HTTPException(404, f"document not found: {body.path}")
        workspace = new_workspace(workspaces)
        try:
            project = open_document(target, workspace)
        except (DocumentError, KeyError, ValueError):
            shutil.rmtree(workspace, ignore_errors=True)
            raise
        return opened(register(workspace, target), project, converted_from=converted_from)

    @app.post("/api/documents/{doc_id}/save")
    def save(doc_id: str, body: SaveBody) -> dict[str, Any]:
        doc = document(doc_id)
        target = inside_projects(body.path, LPHC_SUFFIX) if body.path else doc.path
        if target is None:
            raise HTTPException(400, "choose a document name")
        project = save_document(doc.workspace, body.project, target)
        doc.path = target
        return opened(doc, project)

    @app.delete("/api/documents/{doc_id}")
    def close(doc_id: str) -> dict[str, Any]:
        doc = documents.pop(doc_id, None)
        if doc is not None:
            shutil.rmtree(doc.workspace, ignore_errors=True)
        return {"closed": doc is not None}

    @app.post("/api/documents/{doc_id}/attach")
    def attach(doc_id: str, body: AttachBody) -> dict[str, Any]:
        document(doc_id)
        return {"project": attach_solution(body.project, body.entry)}

    @app.post("/api/documents/{doc_id}/clear")
    def clear(doc_id: str, body: ClearBody) -> dict[str, Any]:
        document(doc_id)
        return {"project": clear_solution(body.project, body.calculation_id)}

    # ---------------------------------------------------------- calculations
    @app.post("/api/calculations/add")
    def calc_add(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        entry = add_calculation(project, name=body.name or forms.OPERATIONS.get(body.operation, "Study"), operation=body.operation)
        project["selected_node"] = {"kind": "study", "id": entry["id"]}
        return {"project": project, "calculation": entry}

    @app.post("/api/calculations/copy")
    def calc_copy(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        entry = copy_calculation(project, body.calculation_id or "")
        project["selected_node"] = {"kind": "study", "id": entry["id"]}
        return {"project": project, "calculation": entry}

    @app.post("/api/calculations/rename")
    def calc_rename(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        rename_calculation(project, body.calculation_id or "", body.name or "")
        return {"project": project}

    @app.post("/api/calculations/delete")
    def calc_delete(body: CalculationBody) -> dict[str, Any]:
        project = clear_solution(body.project, body.calculation_id or "")
        project = dict(delete_calculation(project, body.calculation_id or ""))
        project["plot_groups"] = [group for group in project.get("plot_groups", []) if group.get("calculation_id") != body.calculation_id]
        project["selected_node"] = {"kind": "study", "id": project["calculations"][0]["id"]}
        return {"project": project}

    @app.post("/api/calculations/normalize")
    def calc_normalize(parameters: dict[str, Any]) -> dict[str, Any]:
        return forms.normalize_calculation(parameters)

    # ------------------------------------------------- global definitions / model
    @app.post("/api/parameters/apply")
    def parameters_apply(body: ProjectBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        values = variables(project.get("parameters", []))
        project["model"] = forms.reevaluate_model(project["model"], values)
        return {"project": project, "values": values}

    @app.post("/api/model/form")
    def model_form(body: ModelFormBody) -> dict[str, Any]:
        return forms.model_form(body.model, body.ui_state)

    @app.post("/api/model/apply")
    def model_apply(body: ModelFormBody) -> dict[str, Any]:
        model = forms.apply_model_form(body.model, body.form, variables(body.parameters))
        state = {**body.ui_state, "length_unit": body.form.get("length_unit", "nm"), "material_representation": body.form.get("material_representation", "n")}
        return {"model": model, "form": forms.model_form(model, state)}

    @app.post("/api/sites/rows")
    def sites_rows(body: SitesBody) -> dict[str, Any]:
        return {"rows": forms.site_rows(body.model, body.representation), "cell_center": forms.cell_center(body.model)}

    @app.post("/api/sites/add")
    def sites_add(body: SitesBody) -> dict[str, Any]:
        return {"rows": forms.add_site_row(body.model, body.rows, body.representation, variables(body.parameters))}

    @app.post("/api/sites/apply")
    def sites_apply(body: SitesBody) -> dict[str, Any]:
        model = forms.apply_site_rows(body.model, body.rows, body.representation, variables(body.parameters))
        return {"model": model, "rows": forms.site_rows(model, body.representation)}

    @app.post("/api/preview")
    def preview(body: PreviewBody) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for view in body.views:
            data = preview_geometry(body.model, view=view, size=int(body.size))
            if view not in {"lattice_sites", "reciprocal_bz"}:
                data.pop("points", None)
            data.pop("motif_mask", None)
            result[view] = to_json(data)
        return result

    # ----------------------------------------------------------------- runs
    @app.post("/api/documents/{doc_id}/runs")
    async def start_run(doc_id: str, body: ProjectBody) -> dict[str, Any]:
        doc = document(doc_id)
        project_path = write_workspace_project(doc.workspace, body.project)
        request = build_worker_request(body.project, project_path)
        try:
            run = registry.start(request, project_path)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run.id}

    @app.get("/api/runs/{run_id}")
    def run_state(run_id: str) -> dict[str, Any]:
        run = registry.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return {"run_id": run.id, "state": run.state, "events": to_json(run.events)}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        run = registry.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")

        async def stream():
            sent = 0
            while True:
                while sent < len(run.events):
                    yield f"data: {json.dumps(to_json(run.events[sent]))}\n\n"
                    sent += 1
                if run.state != "running" and sent >= len(run.events) and run.events and run.events[-1].get("event") in {"finished", "cancelled"}:
                    return
                if await request.is_disconnected():
                    return
                run.changed.clear()
                try:
                    await asyncio.wait_for(run.changed.wait(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.delete("/api/runs/{run_id}")
    def cancel_run(run_id: str) -> dict[str, Any]:
        if run_id not in registry.runs:
            raise HTTPException(404, "unknown run")
        run = registry.cancel(run_id)
        return {"run_id": run.id, "state": run.state}

    # -------------------------------------------------------------- records
    def record_directory(doc: OpenDocument, record: str) -> Path:
        records = project_records_dir(doc.workspace / "project.json").resolve()
        target = (doc.workspace / record).resolve()
        if records not in target.parents or not (target / "config.json").is_file():
            raise HTTPException(404, "result record not found in this document")
        return target

    @app.post("/api/documents/{doc_id}/records/view")
    def view_record(doc_id: str, body: RecordBody) -> dict[str, Any]:
        directory = record_directory(document(doc_id), body.record)
        style = body.style
        view = record_view(directory, frequency_unit=style.get("frequency_unit", "Normalized"), component_index=int(style.get("component_index", 0)), field_quantity=style.get("field_quantity", "energy_density"))
        # Rows already carry every inspected value; only small geometric
        # arrays (outlines, centres, q-points) are needed for drawing.
        arrays = real_arrays({name: value for name, value in view.arrays.items() if np.asarray(value).size <= MAX_VIEW_ARRAY})
        extra: dict[str, Any] = {}
        if view.operation in {"band_structure", "frequency_at_k", "efs"}:
            saved = view.config.get("config", view.config)
            length = saved.get("case", {}).get("actual_lattice_constant_m")
            extra["frequency_factor"] = frequency_factor(style.get("frequency_unit", "Normalized"), length)
        if view.operation == "valley_chern" and "cell_polygons" in view.arrays:
            polygons = np.asarray(view.arrays["cell_polygons"], dtype=float)
            extra["sample_cells"] = to_json([polygon[~np.isnan(polygon[:, 0])] for polygon in polygons])
            extra["valley"] = to_json({"valley": view.summary.get("valley"), "bands": view.summary.get("bands", []), "exclusions": view.arrays.get("exclusions")})
        if view.operation == "berry" and "domain_outline" in view.arrays:
            centers = np.asarray(view.arrays.get("sample_centers", np.mean(view.arrays["plaquettes"], axis=1)), dtype=float)
            extra["sample_cells"] = to_json(sample_cell_polygons(centers, np.asarray(view.arrays["domain_outline"], dtype=float)))
        return {
            "operation": view.operation,
            "summary": to_json(view.summary),
            "config": to_json(view.config),
            "columns": list(view.columns),
            "rows": [{"x": row.x, "y": row.y, "values": to_json(row.values)} for row in view.rows],
            "arrays": arrays,
            **extra,
        }

    @app.post("/api/documents/{doc_id}/records/export")
    def export_record(doc_id: str, body: RecordBody) -> Response:
        if body.fmt not in {"png", "svg", "pdf"}:
            raise HTTPException(400, "format must be png, svg or pdf")
        directory = record_directory(document(doc_id), body.record)
        style = {**body.style}
        figure = plot_record(directory, style)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / f"figure.{body.fmt}"
            export_figure(figure, target, width_px=int(style.get("width_px", 900)), height_px=int(style.get("height_px", 600)), dpi=int(style.get("dpi", 100)))
            content = target.read_bytes()
        media = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[body.fmt]
        return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{directory.name}.{body.fmt}"'})

    # -------------------------------------------------------------- presets
    @app.get("/api/presets")
    def list_presets() -> list[dict[str, Any]]:
        if not presets.is_dir():
            return []
        return [{"path": item.name, "name": item.name[: -len(PRESET_SUFFIX)]} for item in sorted(presets.glob(f"*{PRESET_SUFFIX}"))]

    @app.post("/api/presets/save")
    def preset_save(body: PresetBody) -> dict[str, Any]:
        name = "".join(ch for ch in body.name if ch.isalnum() or ch in " _-").strip() or "Untitled"
        preset = new_preset(name)
        preset["parameters"] = {"case": body.model, "calculation": body.calculation, "ui_state": body.ui_state}
        save_preset(presets / f"{name}{PRESET_SUFFIX}", preset)
        return {"path": f"{name}{PRESET_SUFFIX}"}

    @app.post("/api/presets/apply")
    def preset_apply(body: ProjectBody, preset: str) -> dict[str, Any]:
        target = (presets / preset).resolve()
        if target.parent != presets.resolve() or not target.is_file():
            raise HTTPException(404, "preset not found")
        return {"project": dict(apply_preset(body.project, load_preset(target)))}

    # --------------------------------------------------------------- static
    if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets", check_dir=False), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> Response:
            if full_path.startswith("api/"):
                raise HTTPException(404)
            candidate = (STATIC_DIR / full_path).resolve()
            if full_path and candidate.is_file() and STATIC_DIR in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def missing_frontend() -> Response:
            return Response("LegumePhC Studio frontend is not built. Run `npm run build` in web/.", media_type="text/plain")

    return app
