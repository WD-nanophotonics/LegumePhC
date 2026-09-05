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

