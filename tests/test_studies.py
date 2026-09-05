import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("triangular", "square", "affine")


def test_study_modules_import_without_solving():
    for name in STUDIES:
        module = importlib.import_module(f"studies.{name}.study")
        assert callable(module.run)


def test_study_configs_round_trip_and_band_semantics():
    for name in STUDIES:
        path = ROOT / "studies" / name / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["study"] == name
        assert isinstance(config["berry_bands_zero_based"], list)
        assert "zero-based" in (ROOT / "studies" / name / "study.py").read_text(encoding="utf-8")
        assert json.loads(json.dumps(config, sort_keys=True)) == config


def test_studies_route_through_public_apis_and_avoid_forbidden_runtime_dependencies():
    required = ("solve_bands", "frequency_at_k", "compute_field_observables", "solve_efs", "solve_berry")
    forbidden = ("wsl", "mpb", "meep", "thin flow", "courier")
    for name in STUDIES:
        source = (ROOT / "studies" / name / "study.py").read_text(encoding="utf-8").lower()
        assert all(token.lower() in source for token in required)
        assert not any(token in source for token in forbidden)
