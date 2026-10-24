"""Valley Chern numbers over a truncated half Brillouin zone (HBZ).

Construction (Dai, Yoda, Moritake, Notomi, "Fine structures of Berry curvature
and unquantized valley Chern numbers in valley photonic crystals", Sec. B and
App. B; carried over from MePhC E9F, see ``docs/legacy/valley_chern_e9f.md``):

* The K-valley HBZ is the triangle (Γ, b₁, b₂) centred on K=(b₁+b₂)/3; the
  K′ HBZ is its negation.  For LegumePhC's triangular basis K=(2/3, 0).
* The outer boundary is that triangle shrunk about the valley centre so its
  circumradius drops by ``delta_k``.
* Three regular hexagons of circumradius ``delta_gamma`` centred at the
  Γ-equivalent triangle vertices (one hexagon vertex on +qₓ) are excluded.
* Integration uses the integer grid q=(i, j)/N (N=36 in the source):
  C_v = Σ Ω_q w_q / (2π), with two estimators on the same Ω samples:
  ``SOURCE_GRID_MIDPOINT`` (primary; the paper's and the legacy E9F rule) —
  nodes strictly inside the truncated domain, each weighted by 1/N²;
  ``CLIPPED_CELL`` (cross-check) — every node whose 1/N cell overlaps the
  domain, weighted by the exactly clipped cell area.

All coordinates are LegumePhC reduced Cartesian q (2π omitted), so Ω_q and the
cell weights are in the same units and no extra Jacobian appears.  Curvature
is the raw, unsymmetrized rank-1 Wilson flux of each band on a small
counter-clockwise square around each node, divided by its area — the
:func:`legumephc.diagnostics.rank1_wilson` convention (+arg Π⟨uᵢ|uᵢ₊₁⟩ / area),
which is also MePhC's.  A band's valley Chern number is reported only when
*every* sample of that band qualifies (gap, link and branch gates);
otherwise it is ``INCOMPLETE_NOT_REPORTED`` with no number, never a
zero-filled or renormalized partial sum.

Only models with a verified C3 point group have valleys; generic affine
models are rejected rather than assigned a symmetry they do not have.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .berry import _gap_margin, _qualification, _unique_points
from .diagnostics import rank1_wilson
from .geometry import point_group_operations
from .records import create_model_record
from .solver import solve_bands


VALLEYS = ("K", "K'")
EPS = 1e-12


# ---------------------------------------------------------------- polygons
def _area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _ccw(polygon: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=float)
    return polygon if _area(polygon) > 0 else polygon[::-1]


def _cross(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman intersection of a polygon with a convex polygon."""

    output = [np.asarray(p, dtype=float) for p in subject]
    clip = _ccw(clip)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            break
        current, output = output, []
        for p, q in zip(current, current[1:] + current[:1]):
            p_in, q_in = _cross(a, b, p) >= -EPS, _cross(a, b, q) >= -EPS
            if p_in:
                output.append(p)
            if p_in != q_in:
                denominator = _cross(a, b, p) - _cross(a, b, q)
                if abs(denominator) > EPS:
                    t = _cross(a, b, p) / denominator
                    output.append(p + t * (q - p))
    return np.asarray(output, dtype=float).reshape(-1, 2)


def regular_polygon(center: np.ndarray, radius: float, sides: int, rotation: float = 0.0) -> np.ndarray:
    angles = rotation + 2.0 * math.pi * np.arange(sides) / sides
    return np.asarray(center, dtype=float) + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _inside(polygon: np.ndarray, point: np.ndarray) -> bool:
    polygon = _ccw(polygon)
    return all(_cross(a, b, point) >= -EPS for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))


# ------------------------------------------------------------------ domain
@dataclass(frozen=True)
class ValleyDomain:
    valley: str
    center: np.ndarray
    delta_k: float
    delta_gamma: float
    grid_denominator: int
    untruncated: np.ndarray
    outer: np.ndarray
    exclusions: tuple[np.ndarray, ...]
    grid_indices: np.ndarray
    cell_centers: np.ndarray
    cell_weights: np.ndarray
    cell_polygons: tuple[np.ndarray, ...]
    continuous_area: float
    retained: np.ndarray

    @property
    def discrete_area(self) -> float:
        return float(np.sum(self.cell_weights))


def valley_domain(lattice, valley: str = "K'", *, delta_k: float = 0.10, delta_gamma: float = 0.10, grid_denominator: int = 36) -> ValleyDomain:
    """Build the truncated valley HBZ and its clipped-cell quadrature."""

    if valley not in VALLEYS:
        raise ValueError("valley must be 'K' or \"K'\"")
    if delta_k < 0 or delta_gamma < 0 or int(grid_denominator) < 1:
        raise ValueError("delta_k and delta_gamma must be non-negative; grid_denominator positive")
    reciprocal = np.asarray(lattice.reciprocal_basis_reduced, dtype=float)
    b1, b2 = reciprocal[:, 0], reciprocal[:, 1]
    lengths = np.linalg.norm(reciprocal, axis=0)
    cosine = float(b1 @ b2 / (lengths[0] * lengths[1]))
    if not (math.isclose(lengths[0], lengths[1], rel_tol=1e-9) and math.isclose(abs(cosine), 0.5, abs_tol=1e-9)):
        raise ValueError("valley HBZ requires an undeformed triangular reciprocal lattice")
    if cosine < 0:  # use the pair of reciprocal vectors 60° apart
        b2 = b1 + b2
    sign = 1.0 if valley == "K" else -1.0
    untruncated = sign * np.asarray([np.zeros(2), b1, b2])
    center = untruncated.mean(axis=0)
    radius = float(np.linalg.norm(untruncated[0] - center))
    if delta_k >= radius:
        raise ValueError("delta_k removes the whole HBZ")
    outer = _ccw(center + (untruncated - center) * (radius - delta_k) / radius)
    exclusions = tuple(_ccw(regular_polygon(vertex, delta_gamma, 6, 0.0)) for vertex in untruncated) if delta_gamma > 0 else ()
    continuous = abs(_area(outer)) - sum(abs(_area(clip_convex(hole, outer))) for hole in exclusions)

    n = int(grid_denominator)
    h = 1.0 / n
    lo = np.floor(outer.min(axis=0) * n).astype(int) - 1
    hi = np.ceil(outer.max(axis=0) * n).astype(int) + 1
    indices, centers, weights, polygons = [], [], [], []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            q = np.array([i / n, j / n])
            cell = q + h / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
            clipped = clip_convex(cell, outer)
            weight = abs(_area(clipped)) - sum(abs(_area(clip_convex(clip_convex(cell, outer), hole))) for hole in exclusions)
            if weight > EPS:
                indices.append((i, j))
                centers.append(q)
                weights.append(weight)
                polygons.append(clipped)
    centers_array = np.asarray(centers, dtype=float)
    retained = np.asarray([_inside(outer, q) and not any(_inside(hole, q) for hole in exclusions) for q in centers_array], dtype=bool)
    return ValleyDomain(valley, center, float(delta_k), float(delta_gamma), n, untruncated, outer, exclusions,
                        np.asarray(indices, dtype=int), centers_array, np.asarray(weights, dtype=float),
                        tuple(polygons), float(continuous), retained)


def node_class(domain: ValleyDomain) -> np.ndarray:
    """0 = interior node, 1 = node outside the outer boundary, 2 = node inside a Γ exclusion."""

    labels = np.zeros(len(domain.cell_centers), dtype=int)
    for index, q in enumerate(domain.cell_centers):
        if not _inside(domain.outer, q):
            labels[index] = 1
        elif any(_inside(hole, q) for hole in domain.exclusions):
            labels[index] = 2
    return labels


# ----------------------------------------------------------------- solving
def _plaquettes(centers: np.ndarray, stencil: float) -> np.ndarray:
    half = stencil / 2.0
    offsets = half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)  # counter-clockwise
    return centers[:, None, :] + offsets[None, :, :]


def solve_valley_chern(
    model,
    *,
    gmax: float,
    valley: str = "K'",
    bands: tuple[int, ...] = (0, 1, 2),
    delta_k: float = 0.10,
    delta_gamma: float = 0.10,
    grid_denominator: int = 36,
    stencil: float | None = None,
    numeig: int | None = None,
    pol: str = "te",
    gap_floor: float = 0.02,
    link_floor: float = 0.1,
    branch_floor: float = 0.1,
    convergence_status: str = "NOT_ASSESSED",
    record_root: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Rank-1 valley Chern number of each band over the truncated valley HBZ.

    ``bands`` are zero-based.  ``stencil`` is the side of the square Wilson
    plaquette around each grid node (default: the grid step 1/N, as in the
    source).  ``gap_floor`` is the minimum separation of the band from its
    neighbours over the plaquette corners (0.02 = the legacy E9F policy).
    """

    if model.point_group != "C3":
        raise ValueError("valley Chern numbers need a verified C3 model; this model has no valleys")
    if convergence_status not in {"NOT_ASSESSED", "CONVERGED", "FAILED"}:
        raise ValueError("convergence_status must be NOT_ASSESSED, CONVERGED, or FAILED")
    bands = tuple(int(band) for band in bands)
    if not bands or min(bands) < 0:
        raise ValueError("bands are zero-based and must be non-empty")
    domain = valley_domain(model.effective_lattice, valley, delta_k=delta_k, delta_gamma=delta_gamma, grid_denominator=grid_denominator)
    stencil = float(stencil if stencil is not None else 1.0 / domain.grid_denominator)
    if stencil <= 0:
        raise ValueError("stencil must be positive")
    numeig = int(numeig if numeig is not None else max(bands) + 2)
    if numeig < max(bands) + 2:
        raise ValueError("numeig must include the band above the highest target band (for its gap)")

    if model.basis_policy not in {"native", "circular"} and model.closure_qpoints is None:
        # Close the basis on the C3 orbit of Γ (zero lattice shift): a pure
        # rotation closure, identical for K and K' and for every grid node.
        # The solve_bands default (orbit of the first q point) is meant for
        # high-symmetry orbits and would make the basis depend on node order.
        model = replace(model, closure_qpoints=np.zeros((len(point_group_operations("C3")), 2)))
    plaquettes = _plaquettes(domain.cell_centers, stencil)
    area = stencil * stencil
    unique, inverse = _unique_points(plaquettes)
    options: dict[str, Any] = {"gmax": gmax, "numeig": numeig, "pol": pol}
    if progress is not None:
        options["progress"] = progress
        progress({"phase": "valley domain", "message": f"{valley} HBZ: {len(plaquettes)} cells, {len(unique)} distinct q points"})
    solved = solve_bands(model, unique, **options)
    vectors = np.asarray(solved["eigenvectors"])
    frequencies = np.asarray(solved["frequencies"])

    cells, nb = len(plaquettes), len(bands)
    curvature = np.full((cells, nb), np.nan)
    gaps = np.full((cells, nb), np.nan)
    links = np.full((cells, nb), np.nan)
    branch = np.full((cells, nb), np.nan)
    qualified = np.zeros((cells, nb), dtype=bool)
    for cell in range(cells):
        corners = inverse[cell * 4:(cell + 1) * 4]
        for column, band in enumerate(bands):
            wilson = rank1_wilson([vectors[point, :, band] for point in corners])
            gap = _gap_margin(frequencies[corners], (band,))
            verdict = _qualification(gap, wilson["min_link_magnitude"], wilson["branch_margin"], gap_floor=gap_floor, link_floor=link_floor, branch_floor=branch_floor, rank=1)
            curvature[cell, column] = wilson["phase"] / area
            gaps[cell, column], links[cell, column], branch[cell, column] = gap, wilson["min_link_magnitude"], wilson["branch_margin"]
            qualified[cell, column] = verdict["qualified"]
        if progress is not None and (cell + 1) % 50 == 0:
            progress({"phase": "Wilson loops", "completed": cell + 1, "total": cells, "message": f"cells {cell + 1}/{cells}"})

    source_weights = np.where(domain.retained, 1.0 / domain.grid_denominator ** 2, 0.0)
    estimators = {
        "SOURCE_GRID_MIDPOINT": (domain.retained, source_weights),
        "CLIPPED_CELL": (np.ones(cells, dtype=bool), domain.cell_weights),
    }
    per_band = []
    for column, band in enumerate(bands):
        entry: dict[str, Any] = {"band_zero_based": band, "min_gap": float(np.min(gaps[:, column]))}
        for name, (mask, weights) in estimators.items():
            ok = qualified[mask, column]
            failed = ~ok
            complete = bool(np.all(ok))
            flux = float(np.sum(curvature[mask, column] * weights[mask])) if complete else None
            entry[name] = {
                "status": "QUALIFIED_REPORTED" if complete else "INCOMPLETE_NOT_REPORTED",
                "valley_chern": None if flux is None else flux / (2.0 * math.pi),
                "flux_q": flux,
                "qualified_samples": int(np.sum(ok)),
                "total_samples": int(np.sum(mask)),
                "weight_sum": float(np.sum(weights[mask])),
                "failure_reasons": {
                    "gap": int(np.sum(failed & (gaps[mask, column] <= gap_floor))),
                    "link": int(np.sum(failed & (links[mask, column] <= link_floor))),
                    "branch": int(np.sum(failed & (branch[mask, column] <= branch_floor))),
                },
            }
        primary = entry["SOURCE_GRID_MIDPOINT"]
        entry.update({"status": primary["status"], "valley_chern": primary["valley_chern"]})
        per_band.append(entry)
    summary = {
        "valley": valley,
        "domain": {"delta_k": domain.delta_k, "delta_gamma": domain.delta_gamma, "grid_denominator": domain.grid_denominator,
                   "cells": cells, "continuous_area": domain.continuous_area, "discrete_area": domain.discrete_area,
                   "relative_area_error": abs(domain.discrete_area - domain.continuous_area) / domain.continuous_area},
        "primary_estimator": "SOURCE_GRID_MIDPOINT",
        "retained_nodes": int(np.sum(domain.retained)),
        "stencil": stencil,
        "curvature_convention": "rank1_wilson: +arg prod<u_i|u_i+1> / area on a counter-clockwise plaquette (MePhC convention)",
        "inner_product": "TE plane-wave H coefficients" if pol.lower() == "te" else "plane-wave field coefficients",
        "thresholds": {"gap_floor": gap_floor, "link_floor": link_floor, "branch_margin_floor": branch_floor},
        "convergence_status": convergence_status,
        "bands": per_band,
        "raw_unsymmetrized": True,
        "cutoff": {"gmax": float(gmax), "closed_gvec_count": solved.get("closed_gvec_count"), "basis_policy": model.basis_policy,
                   "closure_seed": None if model.closure_qpoints is None else np.asarray(model.closure_qpoints).tolist()},
        "polarization": pol.lower(),
    }
    polygons = np.full((cells, max(len(p) for p in domain.cell_polygons), 2), np.nan)
    for index, polygon in enumerate(domain.cell_polygons):
        polygons[index, :len(polygon)] = polygon
    output = {
        "summary": summary,
        "domain": domain,
        "bands": np.asarray(bands),
        "sample_centers": domain.cell_centers,
        "grid_indices": domain.grid_indices,
        "cell_weights": domain.cell_weights,
        "cell_polygons": polygons,
        "node_class": node_class(domain),
        "retained": domain.retained,
        "plaquettes": plaquettes,
        "curvature": curvature,
        "qualified": qualified,
        "gaps": gaps,
        "links": links,
        "branch_margins": branch,
        "domain_outline": domain.outer,
        "untruncated_hbz": domain.untruncated,
        "exclusions": np.asarray(domain.exclusions) if domain.exclusions else np.zeros((0, 6, 2)),
        "valley_chern": np.asarray([np.nan if item["valley_chern"] is None else item["valley_chern"] for item in per_band]),
    }
    if record_root is not None:
        arrays = {key: value for key, value in output.items() if isinstance(value, np.ndarray)}
        config = {"gmax": gmax, "valley": valley, "bands": list(bands), "delta_k": delta_k, "delta_gamma": delta_gamma,
                  "grid_denominator": grid_denominator, "stencil": stencil, "numeig": numeig, "polarization": pol.lower(),
                  "gap_floor": gap_floor, "link_floor": link_floor, "branch_floor": branch_floor, "convergence_status": convergence_status}
        output["record_path"] = create_model_record(record_root, model, "valley_chern", config, {"status": "succeeded", **summary}, arrays)
    return output
