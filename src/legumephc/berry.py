from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import matrix_covariance_residual, operator_covariance_residual, rank1_wilson, rankn_wilson, reciprocal_c3_map
from .geometry import first_bz_vertices, point_group_operations
from .records import create_model_record
from .solver import solve_bands


def _gap_margin(frequencies: np.ndarray, bands: tuple[int, ...]) -> float:
    first, last = min(bands), max(bands)
    candidates = []
    if first > 0:
        candidates.append(frequencies[:, first] - frequencies[:, first - 1])
    if last + 1 < frequencies.shape[1]:
        candidates.append(frequencies[:, last + 1] - frequencies[:, last])
    return float(min(np.min(value) for value in candidates)) if candidates else float("inf")


def _normalize_plaquettes(plaquettes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(plaquettes, dtype=float)
    if values.shape == (4, 2):
        values = values[None, ...]
    if values.ndim < 3 or values.shape[-2:] != (4, 2):
        raise ValueError("plaquettes must have shape (..., 4, 2)")
    values = values.reshape(-1, 4, 2)
    if not np.isfinite(values).all():
        raise ValueError("plaquettes must contain finite q points")
    areas = 0.5 * np.sum(values[:, :, 0] * np.roll(values[:, :, 1], -1, axis=1) - values[:, :, 1] * np.roll(values[:, :, 0], -1, axis=1), axis=1)
    if np.any(np.abs(areas) <= 1e-14):
        raise ValueError("every plaquette must have nonzero signed area")
    if np.any(np.sign(areas) != np.sign(areas[0])):
        raise ValueError("all plaquettes must have the same orientation")
    return values, areas


def first_bz_plaquettes(lattice, *, grid_size: int = 3, step: float = 0.02) -> np.ndarray:
    """Return a small batch of counter-clockwise plaquettes inside the first BZ."""
    if int(grid_size) < 1 or float(step) <= 0:
        raise ValueError("grid_size must be positive and step must be positive")
    vertices = first_bz_vertices(lattice)
    lower, upper = np.min(vertices, axis=0), np.max(vertices, axis=0)
    # Candidate centers are selected from a regular grid and retained only when
    # all four corners are inside the convex BZ polygon.
    span = upper - lower
    xs = np.linspace(lower[0] + span[0] / (grid_size + 1), upper[0] - span[0] / (grid_size + 1), int(grid_size))
    ys = np.linspace(lower[1] + span[1] / (grid_size + 1), upper[1] - span[1] / (grid_size + 1), int(grid_size))
    centers = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    def inside(points):
        edges = np.roll(vertices, -1, axis=0) - vertices
        rel = points[:, None, :] - vertices[None, :, :]
        cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
        return np.all(cross >= -1e-10, axis=1)
    offsets = np.asarray([[-step, -step], [-step, step], [step, step], [step, -step]])
    valid = inside((centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)).reshape(-1, 4).all(axis=1)
    selected = centers[valid]
    if len(selected) == 0:
        raise ValueError("no first-BZ plaquettes fit the requested step/grid_size")
    return selected[:, None, :] + offsets[None, :, :]


def _unique_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    lookup: dict[tuple[float, float], int] = {}
    inverse: list[int] = []
    for point in values.reshape(-1, 2):
        key = tuple(np.round(point, 14))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(point)
        inverse.append(lookup[key])
    return np.asarray(points), np.asarray(inverse, dtype=int)


def _loop_covariance(solved: dict[str, Any], qloop: np.ndarray, model, pol: str) -> dict[str, Any]:
    if model.point_group is None or len(qloop) != len(point_group_operations(model.point_group)):
        return {"available": False, "reason": "plaquette is not a verified point-group orbit"}
    generator = point_group_operations(model.point_group)[1]
    edges = []
    for index in range(len(qloop)):
        mapping = reciprocal_c3_map(solved["gvec"], solved["gvec"], qloop[index], qloop[(index + 1) % len(qloop)], rotation_matrix=generator)
        edges.append({
            "basis_matching_residual": mapping["maximum_matching_residual"],
            "matched_fraction": mapping["matched_fraction"],
            "epsilon_matrix_residual": matrix_covariance_residual(solved["eps_inv_mat"], mapping),
            "operator_residual": operator_covariance_residual(solved["eps_inv_mat"], solved["gvec"], qloop[index], qloop[(index + 1) % len(qloop)], mapping) if pol.lower() == "te" else None,
        })
    return {"available": True, "edges": edges}


def _qualification(gap: float, link: float, branch: float, *, gap_floor: float, link_floor: float, branch_floor: float, rank: int) -> dict[str, Any]:
    qualified = gap > gap_floor and link > link_floor and branch > branch_floor
    return {
        "gap": gap,
        "link": link,
        "branch_margin": branch,
        "qualified": bool(qualified),
        "status": "QUALIFIED" if qualified else ("RANK1_WITHHELD" if rank == 1 else "UNQUALIFIED"),
        "thresholds": {"gap_floor": gap_floor, "link_floor": link_floor, "branch_margin_floor": branch_floor},
    }


def solve_berry(
    model,
    plaquettes: np.ndarray,
    *,
    gmax: float,
    bands: tuple[int, ...] = (1, 2),
    rank: int | None = None,
    numeig: int = 4,
    pol: str = "te",
    gap_floor: float = 1e-3,
    link_floor: float = 0.1,
    branch_floor: float = 0.1,
    convergence_status: str = "NOT_ASSESSED",
    record_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute unsymmetrized Wilson phase and curvature for each 4-point loop."""

    plaquettes, areas = _normalize_plaquettes(plaquettes)
    bands = tuple(int(band) for band in bands)
    if rank is None:
        rank = 1 if len(bands) == 1 else len(bands)
    if rank == 1 and len(bands) != 1:
        raise ValueError("rank=1 requires one band")
    if rank != len(bands):
        raise ValueError("rank must equal the number of selected bands")
    if convergence_status not in {"NOT_ASSESSED", "CONVERGED", "FAILED"}:
        raise ValueError("convergence_status must be NOT_ASSESSED, CONVERGED, or FAILED")
    unique_qpoints, inverse = _unique_points(plaquettes)
    solved = solve_bands(model, unique_qpoints, gmax=gmax, numeig=max(numeig, max(bands) + 2), pol=pol)
    vectors = np.asarray(solved["eigenvectors"])
    selected = vectors[:, :, bands]
    phases: list[float] = []
    wilsons: list[dict[str, float]] = []
    qualifications: list[dict[str, Any]] = []
    covariance: list[dict[str, Any]] = []
    for index, plaquette in enumerate(plaquettes):
        point_indices = inverse[index * 4:(index + 1) * 4]
        loop = [selected[point, :, :] for point in point_indices]
        wilson = rank1_wilson([value[:, 0] for value in loop]) if rank == 1 else rankn_wilson(loop)
        link = wilson["min_link_magnitude"] if rank == 1 else wilson["min_singular_value"]
        gap = _gap_margin(np.asarray(solved["frequencies"])[point_indices], bands)
        qualification = _qualification(gap, link, wilson["branch_margin"], gap_floor=gap_floor, link_floor=link_floor, branch_floor=branch_floor, rank=rank)
        phases.append(wilson["phase"])
        wilsons.append(wilson)
        qualifications.append(qualification)
        covariance.append(_loop_covariance(solved, plaquette, model, pol))
    phases_array = np.asarray(phases, dtype=float)
    curvature = phases_array / areas
    qualified = np.asarray([item["qualified"] for item in qualifications], dtype=bool)
    qualified_curvature = np.where(qualified, curvature, np.nan)
    aggregate = {
        "qualified": bool(np.all(qualified)),
        "status": "QUALIFIED" if np.all(qualified) else ("RANK1_WITHHELD" if rank == 1 else "UNQUALIFIED"),
        "gate_status": "PASS" if np.all(qualified) else "FAIL",
        "convergence_status": convergence_status,
        "overall_status": "QUALIFIED" if np.all(qualified) and convergence_status == "CONVERGED" else ("RANK1_WITHHELD" if rank == 1 and not np.all(qualified) else "UNQUALIFIED_CONVERGENCE_NOT_ASSESSED" if convergence_status == "NOT_ASSESSED" else "UNQUALIFIED_CONVERGENCE_FAILED"),
        "plaquette_count": len(plaquettes),
        "per_plaquette": qualifications,
    }
    output = {
        "plaquettes": plaquettes,
        "qpoints": unique_qpoints,
        "frequencies": np.asarray(solved["frequencies"])[inverse].reshape(len(plaquettes), 4, -1),
        "eigenvectors": vectors,
        "phases": phases_array,
        "areas": areas,
        "curvature": curvature,
        "qualified_curvature": qualified_curvature,
        "wilson": wilsons,
        "rank": rank,
        "bands": bands,
        "qualification": aggregate,
        "qualifications": qualifications,
        "association": {"metric": "min_link_magnitude" if rank == 1 else "min_singular_value", "values": [item["link"] for item in qualifications]},
        "covariance": covariance,
        "raw_unsymmetrized": True,
        "cutoff": {"gmax": float(gmax), "seed_gvec_count": solved.get("seed_gvec_count"), "closed_gvec_count": solved.get("closed_gvec_count"), "basis_policy": model.basis_policy},
        "polarization": pol.lower(),
    }
    if record_root is not None:
        create_model_record(record_root, model, "solve_berry", {"gmax": gmax, "bands": bands, "rank": rank, "polarization": pol.lower(), "convergence_status": convergence_status}, {"status": "succeeded", "qualification": aggregate, "raw_unsymmetrized": True}, {key: value for key, value in output.items() if isinstance(value, np.ndarray)})
    return output
