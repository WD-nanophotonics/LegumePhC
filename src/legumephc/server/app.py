"""Localhost-only HTTP backend for the LegumePhC web Studio.

Every route delegates to the toolkit-independent Studio layer
(:mod:`legumephc.studio.project`, ``forms``, ``preview``, ``inspection``,
``plotting`` and ``worker``).  Calculations always run in the exact worker
child process; nothing here re-implements a solver.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..studio import forms
from ..studio.inspection import record_view
from ..studio.plotting import export_figure, plot_record, sample_cell_polygons
from ..studio.preview import preview_geometry
from ..studio.project import (
    PRESET_SUFFIX, PROJECT_SUFFIX, add_calculation, apply_preset, copy_calculation, delete_calculation, load_preset,
    load_project, new_preset, new_project, record_available, rename_calculation, save_preset, save_project,
    validate_project,
)
from ..studio.worker import build_worker_request
from ..units import frequency_factor
from .jsonify import real_arrays, to_json
from .runs import RunRegistry


ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = Path(__file__).resolve().parent / "static"
PREVIEW_VIEWS = ("unit_cell", "motif_array", "lattice_sites", "reciprocal_bz", "epsilon")
MAX_VIEW_ARRAY = 20000


class ProjectBody(BaseModel):
    path: str | None = None
    project: dict[str, Any]


class ModelFormBody(BaseModel):
    model: dict[str, Any]
    form: dict[str, Any] = Field(default_factory=dict)
    ui_state: dict[str, Any] = Field(default_factory=dict)


class SitesBody(BaseModel):
    model: dict[str, Any]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    representation: str = "n"


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
    path: str
    record: str
    style: dict[str, Any] = Field(default_factory=dict)
    fmt: str = "png"


class PresetBody(BaseModel):
    name: str
    model: dict[str, Any]
    calculation: dict[str, Any]
    ui_state: dict[str, Any] = Field(default_factory=dict)


def create_app(projects_dir: str | Path | None = None, *, worker_cwd: str | Path | None = None) -> FastAPI:
    projects = Path(projects_dir or ROOT / "projects").resolve()
    presets = projects / "presets"
    registry = RunRegistry(Path(worker_cwd or ROOT).resolve())
    app = FastAPI(title="LegumePhC Studio", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.projects_dir = projects
    app.state.runs = registry

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        payload: dict[str, Any] = {"detail": str(exc)}
        if isinstance(exc, forms.SiteError):
            payload.update({"site_index": exc.index, "field": exc.field})
        return JSONResponse(status_code=422, content=payload)

    def project_file(path: str | None, *, must_exist: bool = True) -> Path:
        if not path:
            raise HTTPException(400, "save the project first")
        candidate = Path(path)
        target = (candidate if candidate.is_absolute() else projects / candidate).resolve()
        try:
            target.relative_to(projects)
        except ValueError as exc:
            raise HTTPException(403, "projects must live inside the Studio projects folder") from exc
        if not target.name.endswith(PROJECT_SUFFIX):
            target = target.with_name(target.name + PROJECT_SUFFIX)
        if must_exist and not target.is_file():
            raise HTTPException(404, f"project not found: {path}")
        return target

    def relative(path: Path) -> str:
        return str(path.relative_to(projects)).replace("\\", "/")

    def record_directory(body: RecordBody) -> Path:
        target = project_file(body.path)
        project = load_project(target)
        entry = next((item for item in project.get("results", []) if item["record_reference"]["path"] == body.record), None)
        if entry is None:
            raise HTTPException(404, "result is not part of this project")
        available, reason = record_available(target.parent, entry["record_reference"])
        if not available:
            raise HTTPException(409, f"result unavailable: {reason}")
        return target.parent / entry["record_reference"]["path"]

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
            "project_suffix": PROJECT_SUFFIX,
        }

    # -------------------------------------------------------------- projects
    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        if not projects.is_dir():
            return []
        items = sorted(projects.rglob(f"*{PROJECT_SUFFIX}"), key=lambda item: item.stat().st_mtime, reverse=True)
        return [{"path": relative(item), "name": item.name[: -len(PROJECT_SUFFIX)], "modified": item.stat().st_mtime} for item in items]

    @app.post("/api/projects/new")
    def create_project(name: str = "Untitled") -> dict[str, Any]:
        return {"path": None, "project": dict(new_project(name))}

    @app.get("/api/projects/open")
    def open_project(path: str) -> dict[str, Any]:
        target = project_file(path)
        return {"path": relative(target), "project": dict(load_project(target))}

    @app.post("/api/projects/save")
    def save(body: ProjectBody) -> dict[str, Any]:
        target = project_file(body.path or f"{body.project.get('name', 'Untitled')}{PROJECT_SUFFIX}", must_exist=False)
        project = validate_project(body.project)
        save_project(target, project)
        return {"path": relative(target), "project": dict(load_project(target))}

    @app.post("/api/projects/validate")
    def validate(body: ProjectBody) -> dict[str, Any]:
        validate_project(body.project)
        return {"ok": True}

    # ---------------------------------------------------------- calculations
    @app.post("/api/calculations/add")
    def calc_add(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        entry = add_calculation(project, name=body.name or "New calculation", operation=body.operation)
        project["selected_node"] = {"kind": "calculation", "id": entry["id"]}
        return {"project": project, "calculation": entry}

    @app.post("/api/calculations/copy")
    def calc_copy(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        entry = copy_calculation(project, body.calculation_id or "")
        project["selected_node"] = {"kind": "calculation", "id": entry["id"]}
        return {"project": project, "calculation": entry}

    @app.post("/api/calculations/rename")
    def calc_rename(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        rename_calculation(project, body.calculation_id or "", body.name or "")
        return {"project": project}

    @app.post("/api/calculations/delete")
    def calc_delete(body: CalculationBody) -> dict[str, Any]:
        project = deepcopy(body.project)
        project = delete_calculation(project, body.calculation_id or "")
        return {"project": dict(project)}

    @app.post("/api/calculations/normalize")
    def calc_normalize(parameters: dict[str, Any]) -> dict[str, Any]:
        return forms.normalize_calculation(parameters)

    # ---------------------------------------------------------------- model
    @app.post("/api/model/form")
    def model_form(body: ModelFormBody) -> dict[str, Any]:
        return forms.model_form(body.model, body.ui_state)

    @app.post("/api/model/apply")
    def model_apply(body: ModelFormBody) -> dict[str, Any]:
        model = forms.apply_model_form(body.model, body.form)
        return {"model": model, "form": forms.model_form(model, {**body.ui_state, "length_unit": body.form.get("length_unit", "nm"), "material_representation": body.form.get("material_representation", "n")})}

    @app.post("/api/sites/rows")
    def sites_rows(body: SitesBody) -> dict[str, Any]:
        return {"rows": forms.site_rows(body.model, body.representation), "cell_center": forms.cell_center(body.model)}

    @app.post("/api/sites/add")
    def sites_add(body: SitesBody) -> dict[str, Any]:
        return {"rows": forms.add_site_row(body.model, body.rows, body.representation)}

    @app.post("/api/sites/apply")
    def sites_apply(body: SitesBody) -> dict[str, Any]:
        model = forms.apply_site_rows(body.model, body.rows, body.representation)
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
    @app.post("/api/runs")
    async def start_run(body: ProjectBody) -> dict[str, Any]:
        target = project_file(body.path, must_exist=False)
        project = validate_project(body.project)
        save_project(target, project)
        request = build_worker_request(load_project(target), target)
        try:
            run = registry.start(request, target)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"run_id": run.id, "path": relative(target)}

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
    @app.post("/api/records/view")
    def view_record(body: RecordBody) -> dict[str, Any]:
        directory = record_directory(body)
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

    @app.post("/api/records/export")
    def export_record(body: RecordBody) -> FileResponse:
        if body.fmt not in {"png", "svg", "pdf"}:
            raise HTTPException(400, "format must be png, svg or pdf")
        directory = record_directory(body)
        style = {**body.style}
        figure = plot_record(directory, style)
        target = directory / "figures" / f"result.{body.fmt}"
        export_figure(figure, target, width_px=int(style.get("width_px", 900)), height_px=int(style.get("height_px", 600)), dpi=int(style.get("dpi", 100)))
        return FileResponse(target, filename=f"{directory.name}.{body.fmt}")

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
