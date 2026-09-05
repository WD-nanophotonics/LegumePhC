from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .records import create_model_record
from .geometry import first_bz_vertices
from .solver import solve_bands


def _inside_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    # The Voronoi cell is convex and its vertices are counter-clockwise.
    # Cross-product inclusion keeps points on the BZ boundary instead of
    # dropping the upper/right edge through ray-casting tie rules.
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = points[:, None, :] - vertices[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1e-12, axis=1)


def _first_bz_grid(model, grid_size: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    vertices = first_bz_vertices(model.effective_lattice)
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    x = np.linspace(lower[0], upper[0], grid_size)
    y = np.linspace(lower[1], upper[1], grid_size)
    mesh = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1)
    mask = _inside_polygon(mesh.reshape(-1, 2), vertices).reshape(grid_size, grid_size)
    return mesh.reshape(-1, 2), mask, (grid_size, grid_size)


def solve_efs(
    model,
    qpoints: np.ndarray | None = None,
    *,
    gmax: float,
    grid_size: int | None = None,
    bands: tuple[int, ...] | None = None,
    numeig: int = 4,
    pol: str = "te",
    record_root: str | Path | None = None,
) -> dict[str, Any]:
    """Sample bands in reciprocal space and return iso-frequency-ready data."""

    grid_qpoints = None
    inside_bz_mask = None
    grid_shape = None
    if grid_size is not None:
        grid_qpoints, inside_bz_mask, grid_shape = _first_bz_grid(model, grid_size)
        qpoints = grid_qpoints[inside_bz_mask.reshape(-1)]
        sampling_domain = "first_bz_grid"
    elif qpoints is not None:
        qpoints = np.asarray(qpoints, dtype=float)
        sampling_domain = "explicit_sparse_samples"
    else:
        raise ValueError("provide qpoints or grid_size")
    solved = solve_bands(model, qpoints, gmax=gmax, numeig=numeig, pol=pol)
    selected = tuple(range(numeig)) if bands is None else tuple(int(band) for band in bands)
    frequencies = np.asarray(solved["frequencies"])[:, selected]
    output = {
        "qpoints": np.asarray(qpoints, dtype=float),
        "frequencies": frequencies,
        "frequency_samples": frequencies,
        "bands": selected,
        "polarization": pol.lower(),
        "iso_frequency_ready": grid_size is not None,
        "sampling_domain": sampling_domain,
        "grid_qpoints": grid_qpoints,
        "inside_bz_mask": inside_bz_mask,
        "grid_shape": grid_shape,
        "cutoff": {"gmax": float(gmax), "seed_gvec_count": solved.get("seed_gvec_count", solved["gvec"].shape[1]), "closed_gvec_count": solved.get("closed_gvec_count")},
    }
    if record_root is not None:
        create_model_record(record_root, model, "solve_efs", {"gmax": gmax, "bands": selected, "polarization": pol.lower(), "grid_size": grid_size}, {"status": "succeeded", "iso_frequency_ready": output["iso_frequency_ready"], "sampling_domain": sampling_domain, "sample_count": len(qpoints), "grid_shape": grid_shape}, {key: value for key, value in output.items() if isinstance(value, np.ndarray)})
    return output
