"""Single-file ``.lphc`` Studio documents (COMSOL ``.mph``-style).

A document is one zip file holding everything a project needs::

    meta.json                      format, save time, record count
    project.json                   the v2 Studio project (model, studies,
                                   results, plot groups, parameters)
    data/.studio/records/<rec>/    immutable records referenced by results
        config.json summary.json arrays.npz figures/*

Opening unpacks the zip into a private *workspace* directory whose layout is
exactly a classic project folder (``project.json`` + ``data/.studio/records``),
so the worker, record references and hashes work unchanged.  Saving writes
``project.json`` and repacks only the records that results still reference,
atomically, so the file never accumulates orphaned solutions.

Study semantics follow COMSOL: each study keeps one solution.  Re-computing a
study replaces its previous result (:func:`attach_solution`); plot groups are
bound to the study, so they show the new solution automatically.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any
import zipfile

from .forms import OPERATIONS
from .project import (
    PROJECT_SUFFIX, load_project, project_records_dir, record_available, save_project, validate_project,
)


LPHC_SUFFIX = ".lphc"
DOCUMENT_FORMAT = "legumephc-document-v1"
PROJECT_MEMBER = "project.json"
META_MEMBER = "meta.json"
RECORDS_PREFIX = "data/.studio/records/"


class DocumentError(ValueError):
    pass


# ----------------------------------------------------------------- project
def referenced_records(project: dict[str, Any]) -> list[str]:
    return sorted({result["record_reference"]["path"] for result in project.get("results", [])})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_plot_group(project: dict[str, Any], calculation_id: str) -> dict[str, Any]:
    """Return the first plot group bound to a study, creating a default one."""

    groups = project.setdefault("plot_groups", [])
    existing = next((group for group in groups if group.get("calculation_id") == calculation_id), None)
    if existing is not None:
        return existing
    used = {group["id"] for group in groups}
    index = 1
    while f"pg-{index}" in used:
        index += 1
    study = next((item for item in project["calculations"] if item["id"] == calculation_id), {"name": calculation_id})
    group = {"id": f"pg-{index}", "name": f"{study.get('name', calculation_id)} plot", "calculation_id": calculation_id, "style": {}, "pins": []}
    groups.append(group)
    return group


def attach_solution(project: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Make ``entry`` the study's only solution and return the project.

    The replaced records are left in the workspace; :func:`save_document`
    drops every record that no result references.
    """

    project = deepcopy(project)
    calculation_id = entry["calculation_id"]
    kept = [result for result in project.get("results", []) if result.get("calculation_id") != calculation_id]
    used = {result["id"] for result in kept}
    index = 1
    while f"result-{index}" in used:
        index += 1
    entry = {**deepcopy(entry), "id": f"result-{index}", "solved_at": _now()}
    project["results"] = [*kept, entry]
    project["selected_result"] = entry["record_reference"]["path"]
    group = ensure_plot_group(project, calculation_id)
    project["selected_node"] = {"kind": "plot", "id": group["id"]}
    return project


def clear_solution(project: dict[str, Any], calculation_id: str | None = None) -> dict[str, Any]:
    """Drop one study's solution, or every solution when ``calculation_id`` is None."""

    project = deepcopy(project)
    project["results"] = [result for result in project.get("results", []) if calculation_id is not None and result.get("calculation_id") != calculation_id]
    if project.get("selected_result") not in referenced_records(project):
        project["selected_result"] = None
    return project


def normalize_document_project(project: dict[str, Any]) -> dict[str, Any]:
    """Enforce one solution per study, splitting extras into imported studies.

    Legacy projects kept several results per calculation.  Rather than drop
    any of them, every additional result becomes its own study whose
    parameters are that result's calculation snapshot.
    """

    project = deepcopy(dict(project))
    project.setdefault("parameters", [])
    project.setdefault("plot_groups", [])
    by_study: dict[str, list[dict[str, Any]]] = {}
    for result in project.get("results", []):
        by_study.setdefault(result["calculation_id"], []).append(result)
    used_ids = {item["id"] for item in project["calculations"]}
    for calculation_id, results in by_study.items():
        if len(results) <= 1:
            continue
        study = next(item for item in project["calculations"] if item["id"] == calculation_id)
        for extra in results[:-1]:  # the newest stays with the original study
            index = 1
            while f"calc-{index}" in used_ids:
                index += 1
            new_id = f"calc-{index}"
            used_ids.add(new_id)
            raw = PurePosixPath(extra["record_reference"]["path"]).name[:15]  # 20261004T052818
            stamp = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} {raw[9:11]}:{raw[11:13]}" if len(raw) == 15 and raw[8] == "T" else raw
            snapshot = deepcopy(extra["calculation_snapshot"])
            label = OPERATIONS.get(snapshot["operation"], snapshot["operation"])
            project["calculations"].append({"id": new_id, "name": f"{label} · imported {stamp}", "operation": snapshot["operation"], "parameters": snapshot})
            extra["calculation_id"] = new_id
    for result in project.get("results", []):
        group = ensure_plot_group(project, result["calculation_id"])
        if not group.get("style") and result.get("plot"):
            group["style"] = deepcopy(result["plot"])
        if not group.get("pins") and result.get("display_state", {}).get("pins"):
            group["pins"] = list(result["display_state"]["pins"])
    return validate_project(project)


# --------------------------------------------------------------- workspace
def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
            raise DocumentError(f"unsafe path in document: {info.filename}")
        if info.filename not in (PROJECT_MEMBER, META_MEMBER) and not info.filename.startswith(RECORDS_PREFIX):
            raise DocumentError(f"unexpected member in document: {info.filename}")
        members.append(info)
    return members


def open_document(source: str | Path, workspace: str | Path) -> dict[str, Any]:
    """Unpack ``source`` into an empty ``workspace`` and return its project."""

    source, workspace = Path(source), Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise DocumentError("workspace must be empty")
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = _safe_members(archive)
        meta = json.loads(archive.read(META_MEMBER)) if META_MEMBER in archive.namelist() else {}
        if meta.get("format") != DOCUMENT_FORMAT:
            raise DocumentError(f"not a LegumePhC document (format={meta.get('format')!r})")
        for info in members:
            archive.extract(info, workspace)
    project = normalize_document_project(load_project(workspace / PROJECT_MEMBER))
    for result in project["results"]:
        available, reason = record_available(workspace, result["record_reference"])
        if not available:
            raise DocumentError(f"document record {result['record_reference']['path']} is damaged: {reason}")
    return project


def new_workspace(base: str | Path) -> Path:
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="ws-", dir=base))


def write_workspace_project(workspace: str | Path, project: dict[str, Any]) -> Path:
    path = Path(workspace) / PROJECT_MEMBER
    save_project(path, validate_project(project))
    return path


def save_document(workspace: str | Path, project: dict[str, Any], target: str | Path) -> dict[str, Any]:
    """Write ``project`` into the workspace and pack it atomically to ``target``."""

    workspace, target = Path(workspace), Path(target)
    project = normalize_document_project(project)
    write_workspace_project(workspace, project)
    keep = referenced_records(project)
    for path in keep:
        available, reason = record_available(workspace, next(r["record_reference"] for r in project["results"] if r["record_reference"]["path"] == path))
        if not available:
            raise DocumentError(f"cannot save: record {path} is unavailable ({reason})")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            meta = {"format": DOCUMENT_FORMAT, "saved_at": _now(), "name": project.get("name"), "record_count": len(keep), "studies": len(project["calculations"])}
            archive.writestr(META_MEMBER, json.dumps(meta, indent=2, sort_keys=True))
            archive.write(workspace / PROJECT_MEMBER, PROJECT_MEMBER)
            for record in keep:
                directory = workspace / record
                for file in sorted(directory.rglob("*")):
                    if file.is_file():
                        member = file.relative_to(workspace).as_posix()
                        compress = zipfile.ZIP_STORED if file.suffix == ".npz" else zipfile.ZIP_DEFLATED
                        archive.write(file, member, compress_type=compress)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _prune_workspace_records(workspace, keep)
    return project


def _prune_workspace_records(workspace: Path, keep: list[str]) -> None:
    records = project_records_dir(workspace / PROJECT_MEMBER)
    if not records.is_dir():
        return
    wanted = {(workspace / path).resolve() for path in keep}
    for directory in records.iterdir():
        if directory.is_dir() and directory.resolve() not in wanted:
            shutil.rmtree(directory)


def read_document_meta(source: str | Path) -> dict[str, Any]:
    with zipfile.ZipFile(source) as archive:
        return json.loads(archive.read(META_MEMBER))


# ------------------------------------------------------------------ legacy
def convert_legacy_project(source: str | Path, target: str | Path | None = None) -> Path:
    """Import a ``.legumephc-studio.json`` project and its records into a new ``.lphc``.

    The legacy project and its record folder are left untouched.
    """

    source = Path(source)
    if not source.name.endswith(PROJECT_SUFFIX):
        raise DocumentError("legacy projects end with " + PROJECT_SUFFIX)
    target = Path(target) if target is not None else source.with_name(source.name[: -len(PROJECT_SUFFIX)] + LPHC_SUFFIX)
    if target.exists():
        raise DocumentError(f"{target.name} already exists")
    project = load_project(source)
    with tempfile.TemporaryDirectory(prefix="lphc-import-") as temporary:
        workspace = Path(temporary)
        for path in referenced_records(project):
            reference = next(r["record_reference"] for r in project["results"] if r["record_reference"]["path"] == path)
            available, reason = record_available(source.parent, reference)
            if not available:
                raise DocumentError(f"legacy record {path} is unavailable: {reason}")
            shutil.copytree(source.parent / path, workspace / path)
        save_document(workspace, project, target)
    return target
