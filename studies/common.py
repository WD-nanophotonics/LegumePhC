from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_study_config(study: str) -> dict[str, Any]:
    path = ROOT / "studies" / study / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def result_root(study: str) -> Path:
    path = ROOT / "studies" / study / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def band_settings(settings: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    bands = tuple(int(value) for value in settings["berry_bands_zero_based"])
    numeig = int(settings["numeig"])
    if not bands or min(bands) < 0 or max(bands) >= numeig:
        raise ValueError("configured band indices must be zero-based and below numeig")
    target = int(settings["target_band_zero_based"])
    if target < 0 or target >= numeig:
        raise ValueError("target_band_zero_based must be below numeig")
    if int(settings["berry_rank"]) != len(bands):
        raise ValueError("berry_rank must match the configured band count")
    return bands, numeig


def synthetic_dipole_inputs(settings: dict[str, Any]) -> tuple[Any, Any]:
    points = np.asarray(settings["synthetic_dipole"]["qpoints"], dtype=float)
    coefficients = np.asarray(settings["synthetic_dipole"]["curvature_coefficients"], dtype=float)
    if coefficients.shape != (3,):
        raise ValueError("synthetic_dipole curvature_coefficients must be [x_gradient, y_gradient, offset]")
    curvature = coefficients[0] * points[:, 0] + coefficients[1] * points[:, 1] + coefficients[2]
    return points, curvature


def report(study: str, summary: dict[str, Any]) -> int:
    print(json.dumps({"study": study, **summary}, indent=2, sort_keys=True))
    return 0
