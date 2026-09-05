import json

import numpy as np

from legumephc.records import create_record


def test_run_directories_never_overwrite(tmp_path):
    identity = {"model": "Model2D", "geometry": "test", "affine": "identity", "basis": "native", "solver": "test", "operation": "pilot"}
    first = create_record(tmp_path, identity=identity, config={"x": 1}, summary={"ok": True}, arrays={"field": np.arange(3)})
    second = create_record(tmp_path, identity=identity, config={"x": 1}, summary={"ok": True}, arrays={"field": np.arange(3)})
    assert first != second
    assert (first / "arrays.npz").exists()
    assert (first / "figures").is_dir()
    assert not (first / "fields.npz").exists()
    assert (second / "summary.json").exists()
    identity = json.loads((first / "config.json").read_text(encoding="utf-8"))["identity"]
    assert set(("model", "geometry", "affine", "basis", "solver", "operation")) <= identity.keys()
    second_identity = json.loads((second / "config.json").read_text(encoding="utf-8"))["identity"]
    assert identity["cache_identity"] == second_identity["cache_identity"]
