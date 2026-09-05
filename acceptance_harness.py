"""Finite Windows acceptance/soak harness for the canonical LegumePhC core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from legumephc import compute_field_observables, frequency_at_k, solve_bands, solve_berry, solve_efs, square_circle_spec
from legumephc.config import load_benchmark
from legumephc.geometry import Affine2D, Lattice2D, m7_orbit
from legumephc.model import Model2D
from legumephc.studio.preview import preview_geometry
from legumephc.studio.project import new_project


ROOT = Path(__file__).resolve().parent
FORBIDDEN_PROCESS_NAMES = ("wsl.exe", "meep", "mpb")
FORBIDDEN_IMPORT_NAMES = ("wsl", "meep", "mpb")
MATRIX_OPERATION_COUNT = 18
DEFAULT_CYCLE_TIMEOUT_MINUTES = 15.0
MAX_CYCLE_TIMEOUT_MINUTES = 60.0
_LAST_CHILD_EVIDENCE: dict | None = None


class MatrixFailure(RuntimeError):
    def __init__(self, message: str, results: list[dict]):
        super().__init__(message)
        self.results = results


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _process_snapshot() -> set[tuple[str, int]]:
    if sys.platform != "win32":
        return set()
    powershell = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    completed = subprocess.run([str(powershell), "-NoProfile", "-Command", "Get-Process | ForEach-Object { \"$($_.Id)|$($_.ProcessName)\" }"], check=True, capture_output=True, text=True)
    snapshot = set()
    for line in completed.stdout.splitlines():
        if "|" in line:
            pid, name = line.split("|", 1)
            snapshot.add((name.strip().lower(), int(pid.strip())))
    return snapshot


def _process_check(baseline: set[tuple[str, int]], allowed_processes: set[tuple[str, int]] | None = None) -> dict:
    current = _process_snapshot()
    allowed = allowed_processes or set()
    introduced = sorted(name for name, pid in current - baseline - allowed if any(name == token or name == token.removesuffix(".exe") or token in name for token in FORBIDDEN_PROCESS_NAMES))
    if introduced:
        raise RuntimeError(f"forbidden process launched by harness: {introduced}")
    baseline_forbidden = sorted(name for name, pid in baseline if any(name == token or name == token.removesuffix(".exe") or token in name for token in FORBIDDEN_PROCESS_NAMES))
    return {"forbidden_introduced": introduced, "forbidden_baseline": baseline_forbidden, "status": "pass"}


def _import_check() -> dict:
    loaded = sorted(name for name in sys.modules if any(name.lower() == token or name.lower().startswith(token + ".") for token in FORBIDDEN_IMPORT_NAMES))
    if loaded:
        raise RuntimeError(f"forbidden imports detected: {loaded}")
    return {"forbidden": loaded, "status": "pass"}


def _record_identity(record: Path | None) -> dict | None:
    if record is None:
        return None
    config = json.loads((record / "config.json").read_text(encoding="utf-8"))
    return config.get("identity")


def _latest_record(records: Path, operation: str) -> Path:
    candidates = [path for path in records.iterdir() if path.is_dir() and f"-{operation}-" in path.name] if records.exists() else []
    if not candidates:
        raise RuntimeError(f"no immutable record for {operation}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _check_berry_record(record: Path) -> dict:
    summary = json.loads((record / "summary.json").read_text(encoding="utf-8"))
    qualification = summary.get("qualification", {})
    overall = qualification.get("overall_status")
    convergence = qualification.get("convergence_status")
    if overall == "QUALIFIED" and convergence != "CONVERGED":
        raise RuntimeError("Berry record is labeled QUALIFIED without CONVERGED evidence")
    if convergence == "NOT_ASSESSED" and overall != "UNQUALIFIED_CONVERGENCE_NOT_ASSESSED":
        raise RuntimeError("Berry convergence-not-assessed record has an invalid overall label")
    return {"overall_status": overall, "gate_status": qualification.get("gate_status"), "convergence_status": convergence}


def _models() -> dict[str, Model2D]:
    benchmark = load_benchmark()
    triangular = Model2D.from_benchmark(benchmark, "G15", lattice=Lattice2D.triangular())
    square = Model2D(square_circle_spec(), Lattice2D.square())
    affine = Model2D(square_circle_spec(), Lattice2D.square(), affine=Affine2D(np.asarray([[1.0, 0.18], [0.0, 0.92]]), np.asarray([0.07, -0.03])))
    return {"triangular": triangular, "square": square, "affine": affine}


def _preview_case(model_name: str) -> dict:
    project = new_project(f"{model_name}-preview")
    case = project["case"]
    if model_name == "triangular":
        case["lattice"] = "triangular"
        case["geometry"]["center"] = [0.0, 0.0]
    elif model_name == "affine":
        case["affine"] = {"linear": [[1.0, 0.18], [0.0, 0.92]], "translation": [0.07, -0.03]}
    data = preview_geometry(case, view="epsilon", size=32)
    return {"shape": list(data["shape"]), "equal_aspect": data["equal_aspect"], "finite": bool(np.isfinite(data["epsilon"]).all())}


def _run_matrix(run_dir: Path, gmax: float = 2.0, index_offset: int = 0, process_baseline: set[tuple[str, int]] | None = None) -> list[dict]:
    records = run_dir / "records"
    records.mkdir(parents=True, exist_ok=True)
    models = _models()
    orbit = m7_orbit(load_benchmark())
    square_q = np.asarray([[0.2, 0.07], [-0.07, 0.2], [-0.2, -0.07], [0.07, -0.2]])
    affine_q = np.asarray([[0.16, 0.09]])
    triangular_plaquette = orbit[0] + np.asarray([[-0.01, -0.01], [-0.01, 0.01], [0.01, 0.01], [0.01, -0.01]])
    operations = [
        ("preview_triangular", lambda: _preview_case("triangular")),
        ("preview_square", lambda: _preview_case("square")),
        ("preview_affine", lambda: _preview_case("affine")),
        ("frequency_triangular", lambda: frequency_at_k(models["triangular"], orbit[0], gmax=gmax, band=1, record_root=records)),
        ("frequency_square", lambda: frequency_at_k(models["square"], square_q[0], gmax=gmax, band=1, record_root=records)),
        ("frequency_affine_te", lambda: frequency_at_k(models["affine"], affine_q[0], gmax=gmax, band=1, pol="te", record_root=records)),
        ("frequency_affine_tm", lambda: frequency_at_k(models["affine"], affine_q[0], gmax=gmax, band=1, pol="tm", record_root=records)),
        ("bands_triangular", lambda: solve_bands(models["triangular"], path="identity", gmax=gmax, numeig=4, record_root=records)),
        ("bands_square", lambda: solve_bands(models["square"], path="identity", gmax=gmax, numeig=4, record_root=records)),
        ("bands_affine_te", lambda: solve_bands(models["affine"], path="identity", gmax=gmax, numeig=3, pol="te", record_root=records)),
        ("bands_affine_tm", lambda: solve_bands(models["affine"], path="identity", gmax=gmax, numeig=3, pol="tm", record_root=records)),
        ("efs_triangular", lambda: solve_efs(models["triangular"], gmax=gmax, grid_size=5, bands=(1, 2), numeig=4, record_root=records)),
        ("efs_square", lambda: solve_efs(models["square"], gmax=gmax, grid_size=5, bands=(1, 2), numeig=4, record_root=records)),
        ("fields_affine_te", lambda: compute_field_observables(solve_bands(models["affine"], affine_q, gmax=gmax, numeig=3, pol="te"), models["affine"], bands=(1,), grid_size=8, record_root=records)),
        ("fields_affine_tm", lambda: compute_field_observables(solve_bands(models["affine"], affine_q, gmax=gmax, numeig=3, pol="tm"), models["affine"], bands=(1,), grid_size=8, record_root=records)),
        ("berry_triangular", lambda: solve_berry(models["triangular"], triangular_plaquette, gmax=gmax, bands=(1, 2), rank=2, numeig=4, record_root=records)),
        ("berry_square", lambda: solve_berry(models["square"], square_q, gmax=gmax, bands=(1, 2), rank=2, numeig=4, record_root=records)),
        ("berry_affine_raw", lambda: solve_berry(models["affine"], affine_q[0] + np.asarray([[-0.02, -0.02], [-0.02, 0.02], [0.02, 0.02], [0.02, -0.02]]), gmax=gmax, bands=(1, 2), rank=2, numeig=4, record_root=records)),
    ]
    results: list[dict] = []
    for index, (name, operation) in enumerate(operations, start=1):
        started = _now()
        started_monotonic = time.monotonic()
        process_before = _process_check(process_baseline or set())
        try:
            value = operation()
            identity = {"operation": "preview" if name.startswith("preview") else "value"}
            record = None
            if not name.startswith("preview"):
                operation_name = {"frequency_triangular": "frequency_at_k", "frequency_square": "frequency_at_k", "frequency_affine_te": "frequency_at_k", "frequency_affine_tm": "frequency_at_k", "bands_triangular": "solve_bands", "bands_square": "solve_bands", "bands_affine_te": "solve_bands", "bands_affine_tm": "solve_bands", "efs_triangular": "solve_efs", "efs_square": "solve_efs", "fields_affine_te": "compute_field_observables", "fields_affine_tm": "compute_field_observables", "berry_triangular": "solve_berry", "berry_square": "solve_berry", "berry_affine_raw": "solve_berry"}[name]
                record = _latest_record(records, operation_name)
                identity = {"path": str(record.relative_to(run_dir)).replace("\\", "/"), "identity": _record_identity(record)}
            if name.startswith("berry"):
                # The harness calls APIs directly; the operation result is still
                # checked for qualification even when no record is requested.
                qualification = value["qualification"]
                overall = qualification["overall_status"]
                if overall == "QUALIFIED" and qualification["convergence_status"] != "CONVERGED":
                    raise RuntimeError("in-memory Berry result was overqualified")
                metadata = {"qualification": {"overall_status": overall, "gate_status": qualification["gate_status"], "convergence_status": qualification["convergence_status"]}, "record_qualification": _check_berry_record(record) if record is not None else None}
            elif isinstance(value, dict) and "frequencies" in value:
                metadata = {"shape": list(np.asarray(value["frequencies"]).shape)}
            elif isinstance(value, dict) and "energy_density" in value:
                metadata = {"shape": list(np.asarray(value["energy_density"]).shape), "gauge_invariant": value.get("gauge_invariant")}
            elif isinstance(value, dict):
                metadata = {key: (list(np.asarray(item).shape) if isinstance(item, np.ndarray) else item) for key, item in value.items() if key in {"shape", "equal_aspect", "finite", "sampling_domain", "iso_frequency_ready"}}
            else:
                metadata = {"value": float(value)}
            process_after = _process_check(process_baseline or set())
            result = {"index": index_offset + index, "name": name, "started": started, "ended": _now(), "duration_seconds": time.monotonic() - started_monotonic, "status": "pass", "result_identity": identity, "metadata": metadata, "process_check": process_after}
        except Exception as exc:
            result = {"index": index_offset + index, "name": name, "started": started, "ended": _now(), "duration_seconds": time.monotonic() - started_monotonic, "status": "fail", "error": f"{type(exc).__name__}: {exc}", "process_check": _process_check(process_baseline or set())}
            results.append(result)
            _write_once(run_dir / "operations" / f"{index_offset + index:05d}-{name}.json", result)
            raise MatrixFailure(f"{type(exc).__name__}: {exc}", results) from exc
        results.append(result)
        _write_once(run_dir / "operations" / f"{index_offset + index:05d}-{name}.json", result)
    return results


def _cycle_result_path(run_dir: Path, cycle_index: int) -> Path:
    return run_dir / "cycle-results" / f"{cycle_index:05d}.json"


def _run_matrix_child(run_dir: Path, gmax: float, index_offset: int, cycle_index: int) -> int:
    cycle_result_path = _cycle_result_path(run_dir, cycle_index)
    cycle_result_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    error = None
    try:
        results = _run_matrix(run_dir, gmax=gmax, index_offset=index_offset, process_baseline=_process_snapshot())
        status = "pass"
    except MatrixFailure as exc:
        results = exc.results
        error = str(exc)
        status = "fail"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "fail"
    cycle_result = {
        "cycle": cycle_index,
        "child_pid": os.getpid(),
        "status": status,
        "operation_count": len(results),
        "results": results,
        "error": error,
    }
    _write_once(cycle_result_path, cycle_result)
    print(json.dumps({"cycle": cycle_index, "child_pid": os.getpid(), "status": status, "operation_count": len(results), "cycle_result": str(cycle_result_path)}, separators=(",", ":"), sort_keys=True))
    return 0 if status == "pass" else 1


def _run_matrix_isolated(run_dir: Path, gmax: float = 2.0, index_offset: int = 0, process_baseline: set[tuple[str, int]] | None = None, cycle_timeout_minutes: float = DEFAULT_CYCLE_TIMEOUT_MINUTES, _popen_fn=subprocess.Popen) -> list[dict]:
    global _LAST_CHILD_EVIDENCE
    if not 0 < cycle_timeout_minutes <= MAX_CYCLE_TIMEOUT_MINUTES:
        raise ValueError(f"cycle_timeout_minutes must be greater than 0 and at most {MAX_CYCLE_TIMEOUT_MINUTES:g}")
    records = run_dir / "records"
    records.mkdir(parents=True, exist_ok=True)
    cycle_index = index_offset // MATRIX_OPERATION_COUNT + 1
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--matrix-child",
        "--run-dir",
        str(run_dir),
        "--gmax",
        f"{gmax:.17g}",
        "--index-offset",
        str(index_offset),
        "--cycle-index",
        str(cycle_index),
    ]
    child = _popen_fn(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_names = {Path(sys.executable).name.lower(), Path(sys.executable).stem.lower()}
    allowed_child = {(name, child.pid) for name in child_names}
    stdout = ""
    stderr = ""
    timed_out = False
    try:
        _process_check(process_baseline or set(), allowed_processes=allowed_child)
        try:
            stdout, stderr = child.communicate(timeout=cycle_timeout_minutes * 60.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                stdout, stderr = child.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.kill()
                stdout, stderr = child.communicate()
    except BaseException:
        if child.poll() is None:
            child.kill()
        stdout, stderr = child.communicate()
        _LAST_CHILD_EVIDENCE = {"cycle": cycle_index, "child_pid": child.pid, "returncode": child.returncode, "timed_out": timed_out, "stdout": stdout, "stderr": stderr, "command": command}
        raise
    _LAST_CHILD_EVIDENCE = {"cycle": cycle_index, "child_pid": child.pid, "returncode": child.returncode, "timed_out": timed_out, "stdout": stdout, "stderr": stderr, "command": command, "cycle_result": str(_cycle_result_path(run_dir, cycle_index))}
    if timed_out:
        raise MatrixFailure(f"isolated matrix child timed out after {cycle_timeout_minutes:g} minutes", [])
    try:
        child_stdout = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MatrixFailure(f"isolated matrix child emitted invalid stdout: {exc}", []) from exc
    cycle_result_path = _cycle_result_path(run_dir, cycle_index)
    cycle_result = json.loads(cycle_result_path.read_text(encoding="utf-8")) if cycle_result_path.exists() else {}
    if child.returncode != 0 or child_stdout.get("status") != "pass" or cycle_result.get("status") != "pass":
        results = cycle_result.get("results", [])
        error = cycle_result.get("error") or f"isolated matrix child exited with code {child.returncode}"
        raise MatrixFailure(error, results)
    if child_stdout.get("child_pid") != cycle_result.get("child_pid") or child_stdout.get("operation_count") != len(cycle_result.get("results", [])):
        raise MatrixFailure("isolated matrix child result identity is inconsistent", cycle_result.get("results", []))
    _process_check(process_baseline or set())
    return cycle_result["results"]


def _record_integrity_check(run_dir: Path) -> dict:
    records = run_dir / "records"
    checked = 0
    for record in records.iterdir() if records.exists() else ():
        if not record.is_dir():
            continue
        config = json.loads((record / "config.json").read_text(encoding="utf-8"))
        json.loads((record / "summary.json").read_text(encoding="utf-8"))
        with np.load(record / "arrays.npz", allow_pickle=False) as stored:
            tuple(stored.files)
        if not config.get("identity", {}).get("operation"):
            raise RuntimeError(f"record identity is incomplete: {record}")
        checked += 1
    return {"status": "pass", "records_checked": checked}


def _wait_until(target: float, clock, sleeper) -> None:
    while True:
        remaining = target - clock()
        if remaining <= 0:
            return
        sleeper(min(60.0, remaining))


def run_harness(*, duration_hours: float | None = None, cycles: int = 1, gmax: float = 2.0, cycle_timeout_minutes: float = DEFAULT_CYCLE_TIMEOUT_MINUTES, _clock=time.monotonic, _sleeper=time.sleep, _matrix_runner=None, _process_snapshot_fn=_process_snapshot, _process_check_fn=_process_check, _import_check_fn=_import_check, _run_dir: Path | None = None) -> Path:
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if duration_hours is not None and duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if not 0 < cycle_timeout_minutes <= MAX_CYCLE_TIMEOUT_MINUTES:
        raise ValueError(f"cycle_timeout_minutes must be greater than 0 and at most {MAX_CYCLE_TIMEOUT_MINUTES:g}")
    run_dir = _run_dir or ROOT / "acceptance" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    (run_dir / "operations").mkdir(parents=True)
    started = _now()
    started_monotonic = _clock()
    all_results: list[dict] = []
    cycle_reports: list[dict] = []
    health_checks: list[dict] = []
    error = None
    interrupted = False
    process_baseline: set[tuple[str, int]] = set()
    import_check: dict = {"status": "not_run"}
    global _LAST_CHILD_EVIDENCE
    _LAST_CHILD_EVIDENCE = None
    matrix_runner = _matrix_runner or (lambda run_dir, *, gmax, index_offset, process_baseline: _run_matrix_isolated(run_dir, gmax=gmax, index_offset=index_offset, process_baseline=process_baseline, cycle_timeout_minutes=cycle_timeout_minutes))
    deadline = None if duration_hours is None else started_monotonic + duration_hours * 3600.0
    try:
        process_baseline = _process_snapshot_fn()
        import_check = _import_check_fn()
        for cycle_index in range(cycles):
            scheduled_offset = 0.0 if deadline is None else (duration_hours * 3600.0) * cycle_index / cycles
            if deadline is not None:
                _wait_until(started_monotonic + scheduled_offset, _clock, _sleeper)
            cycle_started = _now()
            cycle_started_monotonic = _clock()
            cycle_report = {"cycle": cycle_index + 1, "scheduled_offset_seconds": scheduled_offset, "started": cycle_started, "status": "running"}
            try:
                cycle_results = matrix_runner(run_dir, gmax=gmax, index_offset=len(all_results), process_baseline=process_baseline)
                all_results.extend(cycle_results)
                health = {"process": _process_check_fn(process_baseline), "imports": _import_check_fn(), "records": _record_integrity_check(run_dir)}
                health_checks.append({"cycle": cycle_index + 1, "checked": _now(), **health})
                cycle_report.update({"ended": _now(), "duration_seconds": _clock() - cycle_started_monotonic, "status": "pass", "operation_count": len(cycle_results), "health": health})
                if _LAST_CHILD_EVIDENCE is not None:
                    cycle_report["child"] = _LAST_CHILD_EVIDENCE
            except MatrixFailure as exc:
                all_results.extend(exc.results)
                cycle_report.update({"ended": _now(), "duration_seconds": _clock() - cycle_started_monotonic, "status": "fail", "operation_count": len(exc.results), "error": str(exc)})
                if _LAST_CHILD_EVIDENCE is not None:
                    cycle_report["child"] = _LAST_CHILD_EVIDENCE
                cycle_reports.append(cycle_report)
                error = str(exc)
                break
            cycle_reports.append(cycle_report)
        if error is None and deadline is not None:
            while _clock() < deadline:
                health = {"process": _process_check_fn(process_baseline), "imports": _import_check_fn(), "records": _record_integrity_check(run_dir)}
                health_checks.append({"cycle": None, "checked": _now(), **health})
                _wait_until(min(deadline, _clock() + 60.0), _clock, _sleeper)
    except KeyboardInterrupt:
        interrupted = True
        error = "KeyboardInterrupt: interrupted"
    except MatrixFailure as exc:
        all_results.extend(exc.results)
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    report = {
        "schema": "legumephc-phase-e-acceptance-v1",
        "run_id": run_dir.name,
        "started": started,
        "ended": _now(),
        "duration_hours": duration_hours,
        "actual_duration_seconds": _clock() - started_monotonic,
        "cycles_requested": cycles,
        "cycles_completed": len(cycle_reports),
        "interrupted": interrupted,
        "gmax": gmax,
        "operation_count": len(all_results),
        "matrix_passed": error is None,
        "error": error,
        "operations": all_results,
        "cycles": cycle_reports,
        "health_checks": health_checks,
        "import_check": import_check,
        "forbidden_processes": list(FORBIDDEN_PROCESS_NAMES),
        "forbidden_imports": list(FORBIDDEN_IMPORT_NAMES),
        "restart_policy": "Each invocation creates a new immutable run directory; no state is resumed.",
    }
    _write_once(run_dir / "manifest.json", report)
    _write_once(run_dir / "migration_acceptance.json", report)
    print(json.dumps({"run_dir": str(run_dir), "manifest": str(run_dir / 'manifest.json'), "operation_count": len(all_results), "matrix_passed": error is None, "error": error}, indent=2, sort_keys=True))
    if error is not None:
        raise RuntimeError(error)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finite LegumePhC Phase E acceptance/soak harness")
    parser.add_argument("--gmax", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=1, help="number of fixed matrices to run")
    parser.add_argument("--duration-hours", type=float, default=None, help="schedule the fixed cycles across this duration and health-check between cycles")
    parser.add_argument("--cycle-timeout-minutes", type=float, default=DEFAULT_CYCLE_TIMEOUT_MINUTES, help="maximum runtime for each isolated matrix child")
    parser.add_argument("--matrix-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index-offset", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cycle-index", type=int, default=1, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.matrix_child:
        if arguments.run_dir is None:
            parser.error("--matrix-child requires --run-dir")
        return _run_matrix_child(arguments.run_dir, arguments.gmax, arguments.index_offset, arguments.cycle_index)
    run_harness(duration_hours=arguments.duration_hours, cycles=arguments.cycles, gmax=arguments.gmax, cycle_timeout_minutes=arguments.cycle_timeout_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
