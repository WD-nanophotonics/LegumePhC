from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np


PROJECT_SCHEMA_V1 = "legumephc-studio-project-v1"
PROJECT_SCHEMA_V2 = "legumephc-studio-project-v2"
# Kept as the historical name for callers that build v1 worker payloads.
PROJECT_SCHEMA = PROJECT_SCHEMA_V2
PRESET_SCHEMA = "legumephc-studio-preset-v1"
PROJECT_SUFFIX = ".legumephc-studio.json"
PRESET_SUFFIX = ".legumephc-preset.json"


class StudioProject(dict):
    """Canonical v2 mapping with non-serialized v1 adapter views."""
    def __getitem__(self, key):
        if key == "case" and "model" in self:
            return dict.__getitem__(self, "model")
        if key == "calculation" and "calculations" in self:
            return dict.__getitem__(self, "calculations")[0]["parameters"]
        if key == "records" and "results" in self:
            return [item.get("record_reference", item) for item in dict.__getitem__(self, "results")]
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if key == "case" and "model" in self:
            return dict.__setitem__(self, "model", value)
        if key == "calculation" and "calculations" in self:
            dict.__getitem__(self, "calculations")[0]["parameters"] = value
            dict.__getitem__(self, "calculations")[0]["operation"] = value.get("operation", dict.__getitem__(self, "calculations")[0]["operation"])
            return
        if key == "records" and "results" in self:
            dict.__setitem__(self, "results", [{"id": f"result-{index}", "calculation_id": "calc-1", "record_reference": item, "model_snapshot": deepcopy(dict.__getitem__(self, "model")), "calculation_snapshot": deepcopy(dict.__getitem__(self, "calculations")[0]["parameters"]), "plot": deepcopy(dict.__getitem__(self, "plot"))} for index, item in enumerate(value, start=1)])
            return
        dict.__setitem__(self, key, value)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def project_records_dir(project_path: str | Path) -> Path:
    """Return the project-owned immutable record sidecar directory."""

    return Path(project_path).resolve().parent / "data" / ".studio" / "records"


def _identity_affine() -> dict[str, Any]:
    return {"linear": [[1.0, 0.0], [0.0, 1.0]], "translation": [0.0, 0.0]}


def new_project(name: str = "Untitled") -> dict[str, Any]:
    model = {
            "name": "TriangularCircle",
            "lattice": "triangular",
            "lattice_constant": 1.0,
            "direct_basis": [[0.5, 0.5], [0.8660254037844386, -0.8660254037844386]],
            "geometry": {
                "name": "TriangularCircle",
                "kind": "circle",
                "radius": 0.2,
                "sides": 16,
                "angle_degrees": 0.0,
                "center": [0.0, 0.0],
                "epsilon_background": 7.29,
                "epsilon_inclusion": 1.0,
            },
            "affine": _identity_affine(),
            "deformation": {"kind": "none", "factor": 1.0, "angle_degrees": 0.0, "linear": [[1.0, 0.0], [0.0, 1.0]], "translation": [0.0, 0.0]},
            "basis_policy": "auto",
        }
    calculation = {
            "operation": "frequency_at_k",
            "qpoint": [0.2, 0.07],
            "band_one_based": 2,
            "composite_bands_one_based": [2, 3],
            "gmax": 2,
            "numeig": 4,
            "polarization": "te",
            "path": "identity",
            "samples_per_segment": 16,
            "grid_size": 8,
            "efs_grid_size": 5,
            "berry_step": 0.02,
            "sampling_mode": "single_plaquette",
            "centers": [],
            "convergence_status": "NOT_ASSESSED",
            "frequency_window": None,
            "frequency_samples": None,
            "response_weights": None,
            "occupation": None,
            "berry_record_path": None,
        }
    plot = {
            "width_px": 900,
            "height_px": 600,
            "dpi": 100,
            "title": "",
            "x_label": "",
            "y_label": "",
            "grid": True,
            "legend": True,
            "x_limits": None,
            "y_limits": None,
            "band_style": "line",
            "band_line": True,
            "band_markers": False,
            "berry_coloring": False,
            "linewidth": 1.5,
            "marker_size": 4.0,
            "cmap": "viridis",
            "berry_interpolation": False,
            "berry_vmin": None,
            "berry_vmax": None,
            "colorbar": True,
            "component_index": 0,
            "field_quantity": "energy_density",
            "x_limits": None,
            "y_limits": None,
    }
    calculations = [{"id": "calc-1", "name": "Frequency at k", "operation": calculation["operation"], "parameters": calculation}]
    project = StudioProject({
        "schema": PROJECT_SCHEMA_V2,
        "name": str(name),
        "model": model,
        "calculations": calculations,
        "results": [],
        "selected_node": {"kind": "calculation", "id": "calc-1"},
        "selected_result": None,
        "plot": plot,
        "ui_state": {"material_representation": "epsilon", "advanced_expanded": False, "active_geometry_tab": "Motif", "selected_motif": "motif-1"},
    })
    return project


def new_preset(name: str = "Untitled preset") -> dict[str, Any]:
    project = new_project(name)
    return {
        "schema": PRESET_SCHEMA,
        "name": str(name),
        "parameters": {"case": project["case"], "calculation": project["calculation"], "ui_state": deepcopy(project["ui_state"])},
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bind_compatibility_views(project: dict[str, Any]) -> dict[str, Any]:
    """Expose the old single-case views without making them the v2 source."""
    return project if isinstance(project, StudioProject) else StudioProject(project)


def migrate_project(project: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v1 project while preserving its immutable record references."""
    if project.get("schema") != PROJECT_SCHEMA_V1:
        plot = project.setdefault("plot", {})
        legacy = plot.get("band_style", "line")
        plot.setdefault("band_line", legacy != "scatter")
        plot.setdefault("band_markers", legacy in {"scatter", "cycle"})
        return _bind_compatibility_views(project)
    calculation = deepcopy(project["calculation"])
    results = []
    for index, reference in enumerate(project.get("records", []), start=1):
        results.append({"id": f"result-{index}", "calculation_id": "calc-1", "record_reference": deepcopy(reference), "model_snapshot": deepcopy(project["case"]), "calculation_snapshot": deepcopy(calculation), "plot": deepcopy(project.get("plot", {}))})
    selected_id = next((item["id"] for item in results if item["record_reference"].get("path") == project.get("selected_result")), None)
    migrated = {
        "schema": PROJECT_SCHEMA_V2,
        "name": project.get("name", "Untitled"),
        "model": deepcopy(project["case"]),
        "calculations": [{"id": "calc-1", "name": calculation.get("operation", "Calculation"), "operation": calculation.get("operation", "frequency_at_k"), "parameters": calculation}],
        "results": results,
        "selected_node": {"kind": "result", "id": selected_id} if selected_id else {"kind": "calculation", "id": "calc-1"},
        "selected_result": project.get("selected_result"),
        "plot": deepcopy(project.get("plot", {})),
    }
    legacy = migrated["plot"].get("band_style", "line")
    migrated["plot"].setdefault("band_line", legacy != "scatter")
    migrated["plot"].setdefault("band_markers", legacy in {"scatter", "cycle"})
    return _bind_compatibility_views(migrated)


def _validate_calculation(calculation: dict[str, Any]) -> None:
    _require(isinstance(calculation, dict), "project calculation must be an object")
    operation = calculation.get("operation")
    _require(operation in {"frequency_at_k", "band_structure", "fields_energy", "efs", "berry", "berry_curvature_dipole"}, "unsupported calculation operation")
    _require(str(calculation.get("polarization", "")).lower() in {"te", "tm"}, "polarization must be TE or TM")
    _require(int(calculation.get("numeig", 0)) > 0 and float(calculation.get("gmax", 0)) > 0, "calculation cutoff settings must be positive")
    _require(int(calculation.get("band_one_based", 0)) >= 1, "band_one_based must be one-based")
    _require(all(int(value) >= 1 for value in calculation.get("composite_bands_one_based", [])), "composite bands must be one-based")
    if operation == "band_structure":
        _require(int(calculation.get("samples_per_segment", 16)) >= 2, "samples_per_segment must be at least 2")
    if operation in {"fields_energy", "efs"}:
        _require(int(calculation.get("grid_size", calculation.get("efs_grid_size", 0))) >= 2, "grid size must be at least 2")
    if operation == "berry":
        _require(float(calculation.get("berry_step", 0)) > 0, "berry_step must be positive")
        mode = calculation.get("sampling_mode", "single_plaquette")
        _require(mode in {"single_plaquette", "first_bz_grid", "explicit_centers"}, "unsupported Berry sampling mode")
        if mode == "first_bz_grid":
            _require(int(calculation.get("grid_size", 0)) >= 1, "Berry grid_size must be positive")
        if mode == "explicit_centers":
            _require(isinstance(calculation.get("centers", []), list), "Berry centers must be a list")
    if operation == "berry_curvature_dipole":
        if calculation.get("berry_record_path") is not None:
            _require(isinstance(calculation.get("berry_record_path"), str), "BCD Berry record path must be explicit")


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> None:
    array = np.asarray(value, dtype=float)
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(np.isfinite(array).all(), f"{name} must be finite")


def validate_project(project: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(project, dict) and project.get("schema") in {PROJECT_SCHEMA_V1, PROJECT_SCHEMA_V2}, "unsupported Studio project schema")
    if project.get("schema") == PROJECT_SCHEMA_V2:
        for key in ("model", "calculations", "results", "plot", "selected_node"):
            _require(key in project, f"project is missing {key}")
        _require(isinstance(project["calculations"], list) and project["calculations"], "project must contain at least one calculation")
        ids = [item.get("id") for item in project["calculations"]]
        _require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids), "calculation ids must be stable and unique")
        for item in project["calculations"]:
            _require(isinstance(item, dict) and isinstance(item.get("parameters"), dict), "calculation entries need parameters")
            _require(item.get("operation") == item["parameters"].get("operation"), "calculation operation and parameters disagree")
            _validate_calculation(item["parameters"])
        _require(isinstance(project["results"], list), "results must be a list")
        result_ids = [result.get("id") for result in project["results"]]
        _require(len(result_ids) == len(set(result_ids)), "result ids must be stable and unique")
        for result in project["results"]:
            _require(result.get("calculation_id") in ids, "result references an unknown calculation")
            _require(isinstance(result.get("id"), str) and result.get("id"), "result ids must be stable")
            _require(isinstance(result.get("model_snapshot"), dict) and isinstance(result.get("calculation_snapshot"), dict), "results must contain model and calculation snapshots")
            bound = next(item for item in project["calculations"] if item["id"] == result["calculation_id"])
            _require(result["calculation_snapshot"].get("operation") == bound["operation"], "result calculation snapshot is bound to the wrong calculation")
            _require("geometry" in result["model_snapshot"] and "operation" not in result["model_snapshot"], "result model snapshot is malformed")
        case = project["model"]
        calculation = project["calculations"][0]["parameters"]
    else:
        for key in ("case", "calculation", "plot", "records", "selected_result"):
            _require(key in project, f"project is missing {key}")
        case = project["case"]
        calculation = project["calculation"]
    _require(isinstance(case, dict), "project case must be an object")
    _require(case.get("lattice") in {"triangular", "square", "custom"}, "lattice must be triangular, square, or custom")
    lattice_constant = float(case.get("lattice_constant", 1.0))
    _require(np.isfinite(lattice_constant) and lattice_constant > 0, "lattice_constant must be positive and finite")
    geometry = case.get("geometry", {})
    _require(isinstance(geometry, dict), "project geometry must be an object")
    _require(geometry.get("kind") in {"circle", "polygon"}, "geometry kind must be circle or polygon")
    _matrix(case.get("affine", {}).get("linear"), (2, 2), "affine.linear")
    _matrix(case.get("affine", {}).get("translation"), (2,), "affine.translation")
    _require(np.isfinite(float(geometry.get("epsilon_background", 0.0))) and float(geometry.get("epsilon_background", 0.0)) > 0, "epsilon_background must be positive and finite")
    deformation = case.get("deformation")
    if deformation is not None:
        _require(deformation.get("kind") in {"none", "uniaxial", "custom"}, "unsupported deformation kind")
        _matrix(deformation.get("translation", [0.0, 0.0]), (2,), "deformation.translation")
        if deformation.get("kind") == "uniaxial":
            factor = float(deformation.get("factor", 0.0))
            _require(np.isfinite(factor) and factor > 0, "uniaxial factor must be positive and finite")
        elif deformation.get("kind") == "custom":
            _matrix(deformation.get("linear"), (2, 2), "deformation.linear")
    if case["lattice"] == "custom":
        _matrix(case.get("direct_basis"), (2, 2), "direct_basis")
    _matrix(geometry.get("center"), (2,), "geometry.center")
    _require(float(geometry.get("radius", 0.0)) > 0.0, "geometry.radius must be positive")
    if geometry["kind"] == "polygon":
        _require(int(geometry.get("sides", 0)) >= 3, "polygon sides must be at least 3")
    for motif in geometry.get("motifs", []):
        _require(isinstance(motif, dict) and motif.get("kind") in {"circle", "polygon"}, "each motif needs a circle or polygon kind")
        _require(float(motif.get("radius", 0)) > 0, "each motif radius must be positive")
        _matrix(motif.get("center"), (2,), "motif.center")
        if motif["kind"] == "polygon":
            _require(int(motif.get("sides", 0)) >= 3, "motif polygon sides must be at least 3")
        if motif.get("epsilon") is not None:
            _require(float(motif["epsilon"]) > 0, "motif epsilon must be positive")
    _validate_calculation(calculation)
    if project.get("schema") == PROJECT_SCHEMA_V1:
        _require(isinstance(project["records"], list), "records must be a list")
    return project


def validate_preset(preset: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(preset, dict) and preset.get("schema") == PRESET_SCHEMA, "unsupported Studio preset schema")
    _require(set(preset) == {"schema", "name", "parameters"}, "presets contain parameters only and no result references")
    parameters = preset["parameters"]
    _require(isinstance(parameters, dict) and set(parameters).issubset({"case", "calculation", "ui_state"}) and {"case", "calculation"}.issubset(parameters), "preset parameters must contain case and calculation")
    candidate = new_project(preset.get("name", ""))
    candidate["model"] = deepcopy(parameters["case"])
    candidate["calculations"][0]["parameters"] = deepcopy(parameters["calculation"])
    candidate["calculations"][0]["operation"] = parameters["calculation"].get("operation", "frequency_at_k")
    validate_project(candidate)
    return preset


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_project(path: str | Path, project: dict[str, Any]) -> Path:
    validate_project(project)
    target = Path(path)
    _atomic_json(target, project)
    return target


def load_project(path: str | Path) -> dict[str, Any]:
    project = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_project(migrate_project(project))


def save_preset(path: str | Path, preset: dict[str, Any]) -> Path:
    validate_preset(preset)
    target = Path(path)
    _atomic_json(target, preset)
    return target


def load_preset(path: str | Path) -> dict[str, Any]:
    return validate_preset(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_preset(project: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    validate_project(project)
    validate_preset(preset)
    updated = deepcopy(project)
    updated["case"] = deepcopy(preset["parameters"]["case"])
    updated["calculation"] = deepcopy(preset["parameters"]["calculation"])
    if "ui_state" in preset["parameters"]:
        updated["ui_state"] = deepcopy(preset["parameters"]["ui_state"])
    if updated.get("schema") == PROJECT_SCHEMA_V2:
        updated["model"] = updated["case"]
        updated["calculations"][0]["parameters"] = updated["calculation"]
        updated["calculations"][0]["operation"] = updated["calculation"].get("operation", "frequency_at_k")
    return updated


def calculation_by_id(project: dict[str, Any], calculation_id: str) -> dict[str, Any]:
    validate_project(project)
    for calculation in project["calculations"]:
        if calculation["id"] == calculation_id:
            return calculation
    raise KeyError(f"unknown calculation id: {calculation_id}")


def add_calculation(project: dict[str, Any], *, name: str = "New calculation", operation: str = "frequency_at_k", parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_project(project)
    ids = {item["id"] for item in project["calculations"]}
    index = 1
    while f"calc-{index}" in ids:
        index += 1
    base = deepcopy(parameters or project["calculations"][0]["parameters"])
    base["operation"] = operation
    project["calculations"].append({"id": f"calc-{index}", "name": name, "operation": operation, "parameters": base})
    return project["calculations"][-1]


def copy_calculation(project: dict[str, Any], calculation_id: str) -> dict[str, Any]:
    source = calculation_by_id(project, calculation_id)
    return add_calculation(project, name=f"Copy of {source.get('name', 'calculation')}", operation=source["operation"], parameters=source["parameters"])


def rename_calculation(project: dict[str, Any], calculation_id: str, name: str) -> dict[str, Any]:
    calculation = calculation_by_id(project, calculation_id)
    calculation["name"] = str(name).strip() or calculation["id"]
    return calculation


def delete_calculation(project: dict[str, Any], calculation_id: str) -> dict[str, Any]:
    calculation_by_id(project, calculation_id)
    if len(project["calculations"]) <= 1:
        raise ValueError("the project must retain one calculation")
    if any(result.get("calculation_id") == calculation_id for result in project.get("results", [])):
        raise ValueError("cannot delete a calculation referenced by an immutable result")
    project["calculations"] = [item for item in project["calculations"] if item["id"] != calculation_id]
    if project.get("selected_node", {}).get("id") == calculation_id:
        project["selected_node"] = {"kind": "calculation", "id": project["calculations"][0]["id"]}
    _bind_compatibility_views(project)
    return project


def _safe_relative(project_dir: Path, value: str) -> Path:
    candidate = Path(value)
    _require(not candidate.is_absolute(), "record paths must be relative")
    resolved = (project_dir / candidate).resolve()
    try:
        resolved.relative_to(project_dir.resolve())
    except ValueError as exc:
        raise ValueError("record path escapes the project directory") from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_reference(record: str | Path, project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    supplied = Path(record)
    if supplied.is_absolute():
        directory = supplied.resolve()
        try:
            relative = directory.relative_to(root)
        except ValueError as exc:
            raise ValueError("record path escapes the project directory") from exc
    else:
        relative = supplied
        directory = _safe_relative(root, str(relative))
    _require(directory.is_dir(), "record directory does not exist")
    files: dict[str, dict[str, Any]] = {}
    for name in ("config.json", "summary.json", "arrays.npz"):
        file = directory / name
        _require(file.is_file(), f"record is missing {name}")
        files[name] = {"size": file.stat().st_size, "sha256": _sha256(file)}
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    identity = config.get("identity", {})
    return {"path": str(relative).replace("\\", "/"), "files": files, "identity": identity}


def record_available(project_dir: str | Path, reference: dict[str, Any], *, operation: str | None = None) -> tuple[bool, str]:
    try:
        root = Path(project_dir).resolve()
        directory = _safe_relative(root, reference["path"])
        if not directory.is_dir():
            return False, "record directory is missing"
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        identity = config.get("identity", {})
        expected_identity = reference.get("identity", {})
        if not identity.get("model") or not identity.get("operation"):
            return False, "record identity is incomplete"
        for key in ("model", "operation"):
            if expected_identity.get(key) is not None and identity.get(key) != expected_identity[key]:
                return False, f"record {key} identity changed"
        for name, expected in reference["files"].items():
            file = directory / name
            if not file.is_file():
                return False, f"record file {name} is missing"
            if file.stat().st_size != int(expected["size"]):
                return False, f"record file {name} size changed"
            if _sha256(file) != expected["sha256"]:
                return False, f"record file {name} hash changed"
        return True, "available"
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
