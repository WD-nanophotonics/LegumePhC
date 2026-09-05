from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .records import create_record


def compute_berry_dipole(
    qpoints: np.ndarray,
    berry_curvature: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    uncertainty: np.ndarray | float | None = None,
    q_units: str = "reduced",
    reciprocal_basis: np.ndarray | None = None,
    output_units: str = "native",
    frequency_samples: np.ndarray | None = None,
    frequency_window: tuple[float, float] | None = None,
    occupation: np.ndarray | None = None,
    response_weight: np.ndarray | None = None,
    record_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute geometric Berry-curvature gradient and first moment.

    Reduced coordinates produce reduced-coordinate components by default. A
    Cartesian result requires the reciprocal basis explicitly. A physical
    response requires matching frequency samples, a frequency window, and an
    occupation or response weight; otherwise the result remains geometric.
    """

    qpoints = np.asarray(qpoints, dtype=float)
    curvature = np.asarray(berry_curvature, dtype=float).reshape(-1)
    if qpoints.ndim != 2 or qpoints.shape[1] != 2 or len(qpoints) != len(curvature):
        raise ValueError("qpoints must have shape (N,2) matching berry_curvature")
    if q_units not in {"reduced", "cartesian"} or output_units not in {"native", "cartesian"}:
        raise ValueError("q_units must be reduced/cartesian and output_units native/cartesian")
    if output_units == "cartesian" and q_units == "reduced" and reciprocal_basis is None:
        raise ValueError("reciprocal_basis is required for Cartesian output from reduced coordinates")
    reciprocal = None if reciprocal_basis is None else np.asarray(reciprocal_basis, dtype=float)
    if reciprocal is not None and reciprocal.shape != (2, 2):
        raise ValueError("reciprocal_basis must have shape (2,2)")
    coordinates = qpoints
    if q_units == "reduced" and output_units == "cartesian":
        coordinates = qpoints @ reciprocal.T
    weights = np.ones(len(curvature)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if len(weights) != len(curvature) or np.any(weights < 0):
        raise ValueError("weights must be non-negative and match the samples")
    frequencies = None if frequency_samples is None else np.asarray(frequency_samples, dtype=float).reshape(-1)
    if frequencies is not None and len(frequencies) != len(curvature):
        raise ValueError("frequency_samples must match qpoints")
    selected = np.ones(len(curvature), dtype=bool)
    if frequency_window is not None:
        if frequency_window[0] > frequency_window[1]:
            raise ValueError("frequency_window must be increasing")
        if frequencies is not None:
            selected = (frequencies >= frequency_window[0]) & (frequencies <= frequency_window[1])
    if response_weight is not None:
        response_weight = np.asarray(response_weight, dtype=float).reshape(-1)
        if len(response_weight) != len(curvature) or np.any(response_weight < 0):
            raise ValueError("response_weight must be non-negative and match the samples")
        weights = weights * response_weight
    if occupation is not None:
        occupation = np.asarray(occupation, dtype=float).reshape(-1)
        if len(occupation) != len(curvature) or np.any(occupation < 0):
            raise ValueError("occupation must be non-negative and match the samples")
        weights = weights * occupation
    physical = frequencies is not None and frequency_window is not None and (occupation is not None or response_weight is not None)
    if physical:
        weights = np.where(selected, weights, 0.0)
        if np.sum(weights) <= 0:
            raise ValueError("frequency_window and weights select no samples")
    total = float(np.sum(weights))
    first_moment = np.sum(weights[:, None] * coordinates * curvature[:, None], axis=0) / total
    center = np.average(coordinates, axis=0, weights=weights)
    centered_q = coordinates - center
    centered_c = curvature - np.average(curvature, weights=weights)
    normal = (centered_q.T * weights) @ centered_q
    gradient = np.linalg.lstsq(normal, (centered_q.T * weights) @ centered_c, rcond=None)[0]
    if uncertainty is None:
        first_moment_uncertainty = np.zeros(2)
    else:
        sigma = np.broadcast_to(np.asarray(uncertainty, dtype=float), curvature.shape)
        first_moment_uncertainty = np.sqrt(np.sum((weights[:, None] * coordinates / total) ** 2 * sigma[:, None] ** 2, axis=0))
    output: dict[str, Any] = {
        "first_moment": first_moment,
        "gradient": gradient,
        "first_moment_uncertainty": first_moment_uncertainty,
        "q_units": q_units,
        "output_units": output_units,
        "frequency_window": frequency_window,
        "selected_sample_count": int(np.count_nonzero(selected)) if frequency_window is not None and frequencies is not None else len(curvature),
        "window_applied": bool(frequency_window is not None and frequencies is not None),
        "kind": "physical_response" if physical else "geometric",
        "physical_response": physical,
        "weight_source": "occupation" if occupation is not None else ("response_weight" if response_weight is not None else "uniform"),
    }
    if record_root is not None:
        identity = {"model": "BerryDipole", "geometry": "reciprocal-space samples", "affine": "identity", "basis": output_units, "solver": "geometric-core", "operation": "compute_berry_dipole"}
        create_record(record_root, identity=identity, config={"q_units": q_units, "output_units": output_units, "frequency_window": frequency_window}, summary={"status": "succeeded", "kind": output["kind"], "physical_response": physical}, arrays={"qpoints": qpoints, "berry_curvature": curvature, "weights": weights, "first_moment": first_moment, "gradient": gradient, "first_moment_uncertainty": first_moment_uncertainty})
    return output
