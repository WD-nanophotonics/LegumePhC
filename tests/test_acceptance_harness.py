from __future__ import annotations

import json
import os
import subprocess

import acceptance_harness
import pytest


def test_scheduled_soak_uses_exact_cycles_without_real_wait(tmp_path):
    clock_value = [0.0]
    calls = []

    def clock():
        return clock_value[0]

    def sleeper(seconds):
        assert 0.0 < seconds <= 60.0
        clock_value[0] += seconds

    def matrix_runner(run_dir, *, gmax, index_offset, process_baseline):
        calls.append((gmax, index_offset))
        return [{"status": "pass", "name": f"fake-{len(calls)}"}]

    run_dir = acceptance_harness.run_harness(
        duration_hours=0.001,
        cycles=6,
        gmax=2.0,
        _clock=clock,
        _sleeper=sleeper,
        _matrix_runner=matrix_runner,
        _process_snapshot_fn=lambda: set(),
        _process_check_fn=lambda _baseline: {"status": "pass", "forbidden_introduced": []},
        _import_check_fn=lambda: {"status": "pass", "forbidden": []},
        _run_dir=tmp_path / "run",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["matrix_passed"]
    assert manifest["cycles_requested"] == 6
    assert manifest["cycles_completed"] == 6
    assert manifest["operation_count"] == 6
    assert len(calls) == 6
    assert manifest["health_checks"]
    assert manifest["actual_duration_seconds"] >= 3.6


def test_parent_isolates_real_matrices_in_fresh_children(tmp_path):
    run_dir = tmp_path / "run"
    result_dir = acceptance_harness.run_harness(cycles=2, gmax=2.0, _run_dir=run_dir)
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    child_pids = [cycle["child"]["child_pid"] for cycle in manifest["cycles"]]
    current_processes = acceptance_harness._process_snapshot()
    child_names = {os.path.basename(acceptance_harness.sys.executable).lower(), os.path.splitext(os.path.basename(acceptance_harness.sys.executable))[0].lower()}

    assert manifest["matrix_passed"]
    assert manifest["cycles_completed"] == 2
    assert manifest["operation_count"] == 36
    assert all(operation["status"] == "pass" for operation in manifest["operations"])
    assert [operation["index"] for operation in manifest["operations"]] == list(range(1, 37))
    assert len(set(child_pids)) == 2
    assert all(pid != os.getpid() for pid in child_pids)
    assert all(not any(name in child_names and pid == child_pid for name, pid in current_processes) for child_pid in child_pids)
    assert len(list((result_dir / "records").iterdir())) == 30
    assert len(list((result_dir / "operations").glob("*.json"))) == 36
    assert len(list((result_dir / "cycle-results").glob("*.json"))) == 2


def test_isolated_matrix_timeout_terminates_child_and_writes_failed_manifest(tmp_path, monkeypatch):
    class SleepingChild:
        pid = 987654

        def __init__(self, command):
            self.command = command
            self.returncode = None
            self.communicate_calls = []
            self.terminated = False
            self.killed = False

        def communicate(self, timeout=None):
            self.communicate_calls.append(timeout)
            if len(self.communicate_calls) == 1:
                raise subprocess.TimeoutExpired(self.command, timeout, output="partial", stderr="sleeping")
            return "", "terminated"

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def poll(self):
            return self.returncode

    child_holder = {}

    def fake_popen(command, **_kwargs):
        child_holder["child"] = SleepingChild(command)
        return child_holder["child"]

    monkeypatch.setattr(acceptance_harness, "_process_check", lambda *_args, **_kwargs: {"status": "pass"})
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="timed out"):
        acceptance_harness.run_harness(
            cycles=1,
            cycle_timeout_minutes=0.001,
            _matrix_runner=lambda run_dir, *, gmax, index_offset, process_baseline: acceptance_harness._run_matrix_isolated(
                run_dir,
                gmax=gmax,
                index_offset=index_offset,
                process_baseline=process_baseline,
                cycle_timeout_minutes=0.001,
                _popen_fn=fake_popen,
            ),
            _process_snapshot_fn=lambda: set(),
            _process_check_fn=lambda _baseline: {"status": "pass", "forbidden_introduced": []},
            _import_check_fn=lambda: {"status": "pass", "forbidden": []},
            _run_dir=run_dir,
        )

    child = child_holder["child"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert child.terminated
    assert not child.killed
    assert child.returncode == -15
    assert child.poll() is not None
    assert manifest["matrix_passed"] is False
    assert manifest["operation_count"] == 0
    assert manifest["cycles"][0]["child"]["timed_out"] is True
    assert manifest["cycles"][0]["child"]["returncode"] == -15
    assert manifest["cycles"][0]["child"]["stderr"] == "terminated"
