from __future__ import annotations

import json

import acceptance_harness


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
