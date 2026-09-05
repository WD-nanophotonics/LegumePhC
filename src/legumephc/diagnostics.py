from __future__ import annotations

import math
import numpy as np


def relative_orbit_residual(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    scale = max(float(np.max(np.abs(data))), np.finfo(float).eps)
    return float((np.max(data) - np.min(data)) / scale)


def orthonormalize(columns: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(columns, dtype=complex))
    return q


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    ql, qr = orthonormalize(left), orthonormalize(right)
    return float(np.linalg.norm(ql @ ql.conj().T - qr @ qr.conj().T, ord="fro") / math.sqrt(2))


def subspace_singular_values(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    ql, qr = orthonormalize(left), orthonormalize(right)
    return np.linalg.svd(ql.conj().T @ qr, compute_uv=False)


def _unit_phase(value: complex, *, floor: float = 1e-14) -> complex:
    magnitude = abs(value)
    if magnitude <= floor:
        raise ValueError("link magnitude is below numerical floor")
    return value / magnitude


def rank1_wilson(vectors: list[np.ndarray]) -> dict[str, float]:
    normalized = [np.asarray(v, dtype=complex).reshape(-1) for v in vectors]
    normalized = [v / np.linalg.norm(v) for v in normalized]
    overlaps = [np.vdot(normalized[i], normalized[(i + 1) % len(normalized)]) for i in range(len(normalized))]
    phase = float(np.angle(np.prod([_unit_phase(v) for v in overlaps])))
    return {"phase": phase, "min_link_magnitude": float(min(map(abs, overlaps))), "branch_margin": math.pi - abs(phase)}


def rankn_wilson(subspaces: list[np.ndarray]) -> dict[str, float]:
    bases = [orthonormalize(v) for v in subspaces]
    determinants: list[complex] = []
    min_singular = 1.0
    for index, left in enumerate(bases):
        overlap = left.conj().T @ bases[(index + 1) % len(bases)]
        min_singular = min(min_singular, float(np.min(np.linalg.svd(overlap, compute_uv=False))))
        determinants.append(np.linalg.det(overlap))
    phase = float(np.angle(np.prod([_unit_phase(v) for v in determinants])))
    return {"phase": phase, "min_singular_value": min_singular, "branch_margin": math.pi - abs(phase)}


def scalar_field_density(
    eigenvectors: np.ndarray,
    gvec: np.ndarray,
    qpoints: np.ndarray,
    *,
    basis: np.ndarray,
    grid_size: int = 32,
) -> np.ndarray:
    """Evaluate normalized TE H_z densities on a fractional-cell grid.

    The density is periodic, so it is the appropriate scalar observable for a
    C3 comparison; the arbitrary phase of an individual eigenvector cancels.
    ``eigenvectors`` is indexed as (kpoint, reciprocal vector, band).
    """

    vectors = np.asarray(eigenvectors, dtype=complex)
    qpoints = np.asarray(qpoints, dtype=float)
    reciprocal = np.asarray(gvec, dtype=float) / (2.0 * math.pi)
    frac = np.stack(np.meshgrid(
        np.arange(grid_size, dtype=float) / grid_size,
        np.arange(grid_size, dtype=float) / grid_size,
        indexing="xy",
    ), axis=-1).reshape(-1, 2)
    cart = frac @ np.asarray(basis, dtype=float).T
    phases = np.exp(1j * (2.0 * math.pi) * np.einsum(
        "kga,pa->kgp", qpoints[:, None, :] + reciprocal.T, cart
    ))
    fields = np.einsum("kgp,kgb->kpb", phases, vectors)
    density = np.abs(fields) ** 2
    density /= np.mean(density, axis=1, keepdims=True)
    return density.reshape(len(qpoints), grid_size, grid_size, vectors.shape[-1])


def rotated_density_residual(
    densities: np.ndarray,
    qpoints: np.ndarray,
    *,
    basis: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Compare each density with the spatially rotated preceding density."""

    densities = np.asarray(densities, dtype=float)
    qpoints = np.asarray(qpoints, dtype=float)
    inv_basis = np.linalg.inv(np.asarray(basis, dtype=float))
    size = densities.shape[1]
    frac = np.stack(np.meshgrid(
        np.arange(size, dtype=float) / size,
        np.arange(size, dtype=float) / size,
        indexing="xy",
    ), axis=-1).reshape(-1, 2)
    cart = frac @ np.asarray(basis, dtype=float).T
    rotated_frac = (cart @ np.asarray(rotation_matrix, dtype=float)) @ inv_basis.T
    rotated_frac %= 1.0
    rotated_cart = rotated_frac @ np.asarray(basis, dtype=float).T
    result: list[np.ndarray] = []
    for index in range(len(qpoints)):
        previous = (index - 1) % len(qpoints)
        # The scalar density is periodic and can be evaluated directly from
        # its sampled grid.  Coordinates are exact grid points after the
        # 120-degree hexagonal-cell rotation, so nearest-cell indexing is
        # sufficient and avoids interpolation or smoothing.
        frac_indices = np.rint(rotated_frac * size).astype(int) % size
        source = densities[previous, frac_indices[:, 1], frac_indices[:, 0], :]
        target = densities[index].reshape(-1, densities.shape[-1])
        scale = np.maximum(np.linalg.norm(target, axis=0), np.finfo(float).eps)
        result.append(np.linalg.norm(target - source, axis=0) / scale)
    return np.asarray(result)


def reciprocal_basis_projector_residual(
    source: np.ndarray,
    target: np.ndarray,
    source_gvec: np.ndarray,
    target_gvec: np.ndarray,
    q_source: np.ndarray,
    q_target: np.ndarray,
    *,
    rotation_matrix: np.ndarray,
) -> float:
    """Compare subspaces after the exact C3 reciprocal-basis permutation.

    At the M7 orbit, ``q_target - R q_source`` is a reciprocal lattice vector.
    A finite circular g-cut is not invariant under that reciprocal shift, so
    both subspaces are embedded in the union basis and omitted coefficients
    remain visible as truncation error.
    """

    source_g = np.asarray(source_gvec, dtype=float) / (2.0 * math.pi)
    target_g = np.asarray(target_gvec, dtype=float) / (2.0 * math.pi)
    shift = np.asarray(q_target, dtype=float) - np.asarray(rotation_matrix) @ np.asarray(q_source, dtype=float)
    transformed = np.asarray(rotation_matrix) @ source_g - shift[:, None]
    keys = [tuple(np.round(g, 9)) for g in np.column_stack((target_g, transformed)).T]
    union = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    left = np.zeros((len(union), source.shape[1]), dtype=complex)
    right = np.zeros_like(left)
    for index, g in enumerate(transformed.T):
        left[union[tuple(np.round(g, 9))], :] = source[index, :]
    for index, g in enumerate(target_g.T):
        right[union[tuple(np.round(g, 9))], :] = target[index, :]
    return projector_distance(left, right)


def qualification(
    frequencies: np.ndarray,
    rank1_links: list[float],
    rank2_links: list[float],
    rank1_branch_margins: list[float],
    rank2_branch_margins: list[float],
    *,
    rank1_band: int = 1,
    rank2_bands: tuple[int, int] = (1, 2),
    gap_floor: float = 1e-3,
    link_floor: float = 0.1,
    branch_floor: float = 0.1,
) -> dict[str, object]:
    """Apply conservative, explicit gap/link/branch gates."""

    frequencies = np.asarray(frequencies, dtype=float)
    r1 = rank1_band
    r2a, r2b = rank2_bands
    rank1_gap = min(
        float(np.min(frequencies[:, r1] - frequencies[:, r1 - 1])),
        float(np.min(frequencies[:, r1 + 1] - frequencies[:, r1])),
    )
    rank2_gap = min(
        float(np.min(frequencies[:, r2a] - frequencies[:, r2a - 1])),
        float(np.min(frequencies[:, r2b + 1] - frequencies[:, r2b])),
    )
    rank1 = rank1_gap > gap_floor and min(rank1_links) > link_floor and min(rank1_branch_margins) > branch_floor
    rank2 = rank2_gap > gap_floor and min(rank2_links) > link_floor and min(rank2_branch_margins) > branch_floor
    return {
        "thresholds": {"gap_floor": gap_floor, "link_floor": link_floor, "branch_margin_floor": branch_floor},
        "rank1": {
            "gap": rank1_gap,
            "qualified": bool(rank1),
            "status": "QUALIFIED" if rank1 else "RANK1_WITHHELD",
        },
        "rank2": {
            "gap": rank2_gap,
            "qualified": bool(rank2),
            "status": "QUALIFIED" if rank2 else "UNQUALIFIED",
        },
    }
