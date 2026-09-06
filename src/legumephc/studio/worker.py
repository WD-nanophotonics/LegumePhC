from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from copy import deepcopy

import numpy as np

from .. import compute_berry_dipole, compute_field_observables, frequency_at_k, solve_bands, solve_berry, solve_efs
from ..records import create_record
from .profile import model_from_case, zero_based_band, zero_based_bands
from .project import project_records_dir, record_available, validate_project


REQUEST_SCHEMA = "legumephc-studio-worker-request-v1"


def build_worker_request(project: dict[str, Any], project_path: str | Path) -> dict[str, Any]:
    """Build a saved-project request; no run may use an unsaved project."""

    validate_project(project)
    project_file = Path(project_path).resolve()
    if not project_file.is_file():
        raise ValueError("save the project before running a calculation")
    project_dir = project_file.parent
    if project.get("schema", "").endswith("-v2"):
        selected_id = project.get("selected_node", {}).get("id")
        calculation_entry = next((item for item in project["calculations"] if item["id"] == selected_id), project["calculations"][0])
        calculation = calculation_entry["parameters"]
        request = {"schema": REQUEST_SCHEMA, "project_dir": str(project_dir), "records_dir": str(project_records_dir(project_file)), "records": [item["record_reference"] for item in project.get("results", [])], "selected_result": project.get("selected_result"), "case": project["model"], "calculation": calculation, "calculation_id": calculation_entry["id"], "model_snapshot": deepcopy(project["model"]), "calculation_snapshot": deepcopy(calculation)}
    else:
        calculation = project["calculation"]
        request = {"schema": REQUEST_SCHEMA, "project_dir": str(project_dir), "records_dir": str(project_records_dir(project_file)), "records": project.get("records", []), "selected_result": project.get("selected_result"), "case": project["case"], "calculation": calculation}
    validate_request(request)
    return request


def _selected_berry_source(request: dict[str, Any]) -> dict[str, Any]:
    selected = request.get("selected_result") or request["calculation"].get("berry_record_path")
    reference = next((item for item in request.get("records", []) if item.get("path") == selected), None)
    if reference is None or reference.get("identity", {}).get("operation") not in {"berry", "solve_berry"}:
        raise ValueError("BCD requires a selected compatible Berry result")
    project_dir = Path(request["project_dir"]).resolve()
    available, reason = record_available(project_dir, reference)
    if not available:
        raise ValueError(f"selected BCD Berry record is unavailable: {reason}")
    directory = project_dir / Path(reference["path"])
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        with np.load(directory / "arrays.npz", allow_pickle=False) as stored:
            if "plaquettes" in stored.files:
                centers = np.mean(stored["plaquettes"], axis=1)
            elif "qpoints" in stored.files:
                centers = np.asarray(stored["qpoints"])
            else:
                raise ValueError("selected Berry record has no qpoints or plaquettes")
            if "curvature" not in stored.files:
                raise ValueError("selected Berry record has no curvature")
            curvature = np.asarray(stored["curvature"], dtype=float).reshape(-1)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"selected BCD Berry record is unreadable: {exc}") from exc
    status = summary.get("qualification", {}).get("overall_status")
    if status != "QUALIFIED":
        raise ValueError(f"selected Berry record is not production-qualified: overall_status={status}")
    identity = config.get("identity", {})
    if identity.get("model") != "Model2D" or identity.get("operation") not in {"berry", "solve_berry"}:
        raise ValueError("selected Berry record has incompatible model/operation identity")
    if len(centers) != len(curvature):
        raise ValueError("selected Berry record has mismatched plaquette centers and curvature")
    calculation = request["calculation"]
    for name in ("frequency_samples", "response_weights", "occupation"):
        values = calculation.get(name)
        if values is not None and len(values) != len(centers):
            raise ValueError(f"{name} must contain one value per Berry plaquette center ({len(centers)})")
    if calculation.get("frequency_window") is not None:
        if calculation.get("frequency_samples") is None:
            raise ValueError("physical BCD requires frequency_samples for the selected Berry plaquette centers")
        if calculation.get("response_weights") is None and calculation.get("occupation") is None:
            raise ValueError("physical BCD requires response_weights or occupation for the selected Berry plaquette centers")
    return {"reference": reference, "config": config, "summary": summary, "identity": identity, "centers": centers, "curvature": curvature}


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported Studio worker request schema")
    if not isinstance(request.get("project_dir"), str) or not isinstance(request.get("records_dir"), str):
        raise ValueError("worker request must contain project_dir and records_dir")
    project = validate_project({
        "schema": "legumephc-studio-project-v1",
        "name": "worker",
        "case": request.get("case"),
        "calculation": request.get("calculation"),
        "plot": {}, "records": [], "selected_result": None,
    })
    project_dir = Path(request["project_dir"]).resolve()
    records_dir = Path(request["records_dir"]).resolve()
    try:
        records_dir.relative_to(project_dir)
    except ValueError as exc:
        raise ValueError("records_dir must be inside project_dir") from exc
    if project["calculation"]["operation"] == "berry_curvature_dipole":
        _selected_berry_source({**request, "case": project["case"], "calculation": project["calculation"]})
    return {**request, "case": project["case"], "calculation": project["calculation"]}


def _selected_bands(calculation: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    bands = zero_based_bands(calculation["composite_bands_one_based"])
    target = zero_based_band(calculation["band_one_based"])
    numeig = int(calculation["numeig"])
    if max((*bands, target), default=0) >= numeig:
        raise ValueError("configured one-based band exceeds numeig")
    return bands, numeig


def _plaquette(calculation: dict[str, Any]) -> np.ndarray:
    center = np.asarray(calculation["qpoint"], dtype=float)
    step = float(calculation["berry_step"])
    return center + np.asarray([[-step, -step], [-step, step], [step, step], [step, -step]])


def _record_identity(model, operation: str) -> dict[str, Any]:
    return {
        "model": "Model2D",
        "geometry": model.geometry.name,
        "affine": {"linear": model.affine.linear.tolist(), "translation": model.affine.translation.tolist()},
        "basis": model.identity["basis"],
        "solver": "legumephc.public-api",
        "operation": operation,
    }


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(request)
    calculation = request["calculation"]
    operation = calculation["operation"]
    bands, numeig = _selected_bands(calculation)
    gmax = float(calculation["gmax"])
    pol = str(calculation["polarization"]).lower()
    arrays: dict[str, np.ndarray] = {}
    source = _selected_berry_source(request) if operation == "berry_curvature_dipole" else None
    model = None if source is not None else model_from_case(request["case"])
    summary: dict[str, Any] = {"status": "succeeded", "operation": operation}
    if model is not None:
        summary["point_group"] = model.point_group
    if operation == "frequency_at_k":
        qpoint = np.asarray(calculation["qpoint"], dtype=float)
        value = frequency_at_k(model, qpoint, gmax=gmax, band=zero_based_band(calculation["band_one_based"]), pol=pol)
        arrays = {"qpoint": qpoint.reshape(1, 2), "frequency": np.asarray([value])}
        summary.update({"frequency": value, "band_one_based": calculation["band_one_based"], "polarization": pol})
    elif operation == "band_structure":
        result = solve_bands(model, path=calculation.get("path", "identity"), gmax=gmax, numeig=numeig, pol=pol, samples_per_segment=int(calculation.get("samples_per_segment", 16)))
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        summary.update({"path_labels": result["path_labels"], "qpoint_count": len(result["qpoints"]), "polarization": pol})
    elif operation == "fields_energy":
        qpoint = np.asarray(calculation["qpoint"], dtype=float).reshape(1, 2)
        result = solve_bands(model, qpoint, gmax=gmax, numeig=numeig, pol=pol)
        observed = compute_field_observables(result, model, bands=bands, grid_size=int(calculation["grid_size"]))
        arrays = {key: value for key, value in observed.items() if isinstance(value, np.ndarray)}
        summary.update({"bands_zero_based": bands, "grid_size": int(calculation["grid_size"]), "gauge_invariant": True, "polarization": pol})
    elif operation == "efs":
        result = solve_efs(model, gmax=gmax, grid_size=int(calculation["efs_grid_size"]), bands=bands, numeig=numeig, pol=pol)
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        summary.update({"bands_zero_based": bands, "sampling_domain": result["sampling_domain"], "grid_shape": result["grid_shape"], "sample_count": len(result["qpoints"]), "polarization": pol})
    elif operation == "berry":
        sampling_mode = calculation.get("sampling_mode", "single_plaquette")
        if sampling_mode == "first_bz_grid":
            from ..berry import first_bz_plaquettes
            plaquettes = first_bz_plaquettes(model.effective_lattice, grid_size=int(calculation.get("grid_size", 3)), step=float(calculation["berry_step"]))
        elif sampling_mode == "explicit_centers":
            centers = np.asarray(calculation.get("centers", []), dtype=float)
            step = float(calculation["berry_step"])
            offsets = np.asarray([[-step, -step], [-step, step], [step, step], [step, -step]])
            plaquettes = centers[:, None, :] + offsets[None, :, :]
        else:
            plaquettes = _plaquette(calculation)
        result = solve_berry(model, plaquettes, gmax=gmax, bands=bands, rank=len(bands), numeig=numeig, pol=pol, convergence_status=calculation.get("convergence_status", "NOT_ASSESSED"))
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        summary.update({"qualification": result["qualification"], "raw_unsymmetrized": True, "polarization": pol})
    elif operation == "berry_curvature_dipole":
        selected = source["reference"]["path"]
        qpoints = source["centers"]
        curvature = source["curvature"]
        frequency_samples = calculation.get("frequency_samples")
        response_weights = calculation.get("response_weights")
        occupation = calculation.get("occupation")
        result = compute_berry_dipole(qpoints, curvature, q_units="reduced", frequency_samples=frequency_samples, frequency_window=calculation.get("frequency_window"), response_weight=response_weights, occupation=occupation)
        arrays = {"qpoints": qpoints, "berry_curvature": curvature, "first_moment": result["first_moment"], "gradient": result["gradient"]}
        if frequency_samples is not None:
            arrays["frequency_samples"] = np.asarray(frequency_samples, dtype=float)
        provenance = {"path": selected, "files": source["reference"]["files"], "identity": source["identity"], "qualification": source["summary"].get("qualification")}
        summary.update({"kind": result["kind"], "physical_response": result["physical_response"], "selected_sample_count": result["selected_sample_count"], "source_berry": provenance})
    else:
        raise ValueError(f"unsupported operation {operation}")
    identity = _record_identity(model, operation) if model is not None else {**source["identity"], "operation": operation}
    record_config = {"case": request["case"], "calculation": calculation}
    if source is not None:
        record_config["source_berry"] = {"path": source["reference"]["path"], "files": source["reference"]["files"], "identity": source["identity"], "qualification": source["summary"].get("qualification")}
    record = create_record(request["records_dir"], identity=identity, config=record_config, summary=summary, arrays=arrays)
    return {"record_path": str(record.resolve()), "metadata": {"operation": operation, "summary": summary}}


def _child_main(request_path: str) -> int:
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        result = execute_request(request)
        print(json.dumps({"ok": True, **result}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:  # the parent receives a small structured failure
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), flush=True)
        return 1


class WorkerProcess:
    """Handle exactly one child process and its temporary request file."""

    def __init__(self, process: subprocess.Popen[str], request_path: Path):
        self.process = process
        self.request_path = request_path

    def poll(self) -> int | None:
        return self.process.poll()

    def communicate(self) -> tuple[str, str]:
        stdout, stderr = self.process.communicate()
        self._cleanup()
        return stdout, stderr

    def cancel(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._cleanup()

    def _cleanup(self) -> None:
        if self.request_path.exists():
            self.request_path.unlink()


def start_worker(request: dict[str, Any], *, python_executable: str | None = None, cwd: str | Path | None = None) -> WorkerProcess:
    request = validate_request(request)
    fd, path_string = tempfile.mkstemp(prefix="legumephc-studio-request-", suffix=".json")
    import os
    os.close(fd)
    request_path = Path(path_string)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    command = [python_executable or sys.executable, "-m", "legumephc.studio.worker", "--request", path_string]
    try:
        process = subprocess.Popen(command, cwd=str(cwd) if cwd is not None else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        request_path.unlink(missing_ok=True)
        raise
    return WorkerProcess(process, request_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LegumePhC Studio exact child worker")
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args(argv)
    return _child_main(arguments.request)


if __name__ == "__main__":
    raise SystemExit(main())
