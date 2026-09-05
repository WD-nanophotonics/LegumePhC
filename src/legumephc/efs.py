from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .records import create_model_record
from .solver import solve_bands


def solve_efs(
    model,
    qpoints: np.ndarray,
    *,
    gmax: float,
    bands: tuple[int, ...] | None = None,
    numeig: int = 4,
    pol: str = "te",
    record_root: str | Path | None = None,
) -> dict[str, Any]:
    """Sample bands in reciprocal space and return iso-frequency-ready data."""

    solved = solve_bands(model, np.asarray(qpoints, dtype=float), gmax=gmax, numeig=numeig, pol=pol)
    selected = tuple(range(numeig)) if bands is None else tuple(int(band) for band in bands)
    frequencies = np.asarray(solved["frequencies"])[:, selected]
    output = {
        "qpoints": np.asarray(qpoints, dtype=float),
        "frequencies": frequencies,
        "frequency_samples": frequencies,
        "bands": selected,
        "polarization": pol.lower(),
        "iso_frequency_ready": True,
        "cutoff": {"gmax": float(gmax), "seed_gvec_count": solved.get("seed_gvec_count", len(solved["gvec"][0]) if solved["gvec"].ndim == 2 else None), "closed_gvec_count": solved.get("closed_gvec_count")},
    }
    if record_root is not None:
        create_model_record(record_root, model, "solve_efs", {"gmax": gmax, "bands": selected, "polarization": pol.lower()}, {"status": "succeeded", "iso_frequency_ready": True, "sample_count": len(qpoints)}, {key: value for key, value in output.items() if isinstance(value, np.ndarray)})
    return output
