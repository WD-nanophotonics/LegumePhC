from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np


PROJECT_SCHEMA = "legumephc-studio-project-v1"
PRESET_SCHEMA = "legumephc-studio-preset-v1"
PROJECT_SUFFIX = ".legumephc-studio.json"
PRESET_SUFFIX = ".legumephc-preset.json"


def project_records_dir(project_path: str | Path) -> Path:
    """Return the project-owned immutable record sidecar directory."""

    return Path(project_path).resolve().parent / "data" / ".studio" / "records"


def _identity_affine() -> dict[str, Any]:
    return {"linear": [[1.0, 0.0], [0.0, 1.0]], "translation": [0.0, 0.0]}


def new_project(name: str = "Untitled") -> dict[str, Any]:
    return {
        "schema": PROJECT_SCHEMA,
        "name": str(name),
        "case": {
            "lattice": "square",
            "lattice_constant": 1.0,
            "direct_basis": [[1.0, 0.0], [0.0, 1.0]],
            "geometry": {
                "name": "SquareCircle",
                "kind": "circle",
                "radius": 0.2,
                "sides": 16,
                "angle_degrees": 0.0,
                "center": [0.5, 0.5],
                "epsilon_background": 7.29,
                "epsilon_inclusion": 1.0,
            },
            "affine": _identity_affine(),
            "basis_policy": "auto",
        },
        "calculation": {
            "operation": "frequency_at_k",
            "qpoint": [0.2, 0.07],
            "band_one_based": 2,
            "composite_bands_one_based": [2, 3],
            "gmax": 2,
            "numeig": 4,
            "polarization": "te",
            "path": "identity",
            "grid_size": 8,
            "efs_grid_size": 5,
            "berry_step": 0.02,
            "frequency_window": None,
            "frequency_samples": None,
            "response_weights": None,
            "occupation": None,
            "berry_record_path": None,
        },
        "plot": {
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
            "berry_coloring": False,
        },
        "records": [],
        "selected_result": None,
    }


def new_preset(name: str = "Untitled preset") -> dict[str, Any]:
    project = new_project(name)
    return {
        "schema": PRESET_SCHEMA,
        "name": str(name),
        "parameters": {"case": project["case"], "calculation": project["calculation"]},
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> None:
    array = np.asarray(value, dtype=float)
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(np.isfinite(array).all(), f"{name} must be finite")


def validate_project(project: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(project, dict) and project.get("schema") == PROJECT_SCHEMA, "unsupported Studio project schema")
    for key in ("case", "calculation", "plot", "records", "selected_result"):
        _require(key in project, f"project is missing {key}")
    case = project["case"]
    _require(isinstance(case, dict), "project case must be an object")
    _require(case.get("lattice") in {"triangular", "square", "custom"}, "lattice must be triangular, square, or custom")
    geometry = case.get("geometry", {})
    _require(isinstance(geometry, dict), "project geometry must be an object")
    _require(geometry.get("kind") in {"circle", "polygon"}, "geometry kind must be circle or polygon")
    _matrix(case.get("affine", {}).get("linear"), (2, 2), "affine.linear")
    _matrix(case.get("affine", {}).get("translation"), (2,), "affine.translation")
    if case["lattice"] == "custom":
        _matrix(case.get("direct_basis"), (2, 2), "direct_basis")
    _matrix(geometry.get("center"), (2,), "geometry.center")
    _require(float(geometry.get("radius", 0.0)) > 0.0, "geometry.radius must be positive")
    if geometry["kind"] == "polygon":
        _require(int(geometry.get("sides", 0)) >= 3, "polygon sides must be at least 3")
    calculation = project["calculation"]
    _require(isinstance(calculation, dict), "project calculation must be an object")
    _require(calculation.get("operation") in {"frequency_at_k", "band_structure", "fields_energy", "efs", "berry", "berry_curvature_dipole"}, "unsupported calculation operation")
    _require(str(calculation.get("polarization", "")).lower() in {"te", "tm"}, "polarization must be TE or TM")
    _require(int(calculation.get("numeig", 0)) > 0 and float(calculation.get("gmax", 0)) > 0, "calculation cutoff settings must be positive")
    _require(int(calculation.get("band_one_based", 0)) >= 1, "band_one_based must be one-based")
    _require(all(int(value) >= 1 for value in calculation.get("composite_bands_one_based", [])), "composite bands must be one-based")
    _require(isinstance(project["records"], list), "records must be a list")
    return project


def validate_preset(preset: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(preset, dict) and preset.get("schema") == PRESET_SCHEMA, "unsupported Studio preset schema")
    _require(set(preset) == {"schema", "name", "parameters"}, "presets contain parameters only and no result references")
    parameters = preset["parameters"]
    _require(isinstance(parameters, dict) and set(parameters) == {"case", "calculation"}, "preset parameters must contain case and calculation")
    validate_project({**new_project(preset.get("name", "")), "case": parameters["case"], "calculation": parameters["calculation"]})
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
    return validate_project(project)


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
    return updated


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
