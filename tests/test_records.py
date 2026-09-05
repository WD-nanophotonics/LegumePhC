import json

import numpy as np

from legumephc.records import create_run


def test_run_directories_never_overwrite(tmp_path):
    first = create_run(tmp_path, "pilot", {"x": 1}, {"ok": True}, {"field": np.arange(3)})
    second = create_run(tmp_path, "pilot", {"x": 1}, {"ok": True}, {"field": np.arange(3)})
    assert first != second
    assert (first / "arrays.npz").exists()
    assert (first / "figures").is_dir()
    assert (first / "fields.npz").exists()
    assert (second / "summary.json").exists()
    identity = json.loads((first / "config.json").read_text(encoding="utf-8"))["identity"]
    assert set(("model", "geometry", "affine", "basis", "solver", "operation")) <= identity.keys()
