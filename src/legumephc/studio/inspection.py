from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..units import frequency_factor, frequency_label
from .profile import model_from_case


OPERATION_ALIASES = {
    "solve_bands": "band_structure",
    "compute_field_observables": "fields_energy",
    "solve_efs": "efs",
    "solve_berry": "berry",
    "compute_berry_dipole": "berry_curvature_dipole",
}


def format_significant(value: float, digits: int = 6) -> str:
    """Format a finite scalar for interactive readout, without float noise."""

    number = float(value)
    if not np.isfinite(number):
        return str(number)
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude < 10 ** (-(digits - 1)) or magnitude >= 10**digits:
        return f"{number:.{digits - 1}e}"
    return f"{number:.{digits}g}"


def nearest_screen_point(points: np.ndarray, cursor: Iterable[float], *, radius_px: float = 10.0) -> int | None:
    """Return the closest row in already-transformed screen coordinates."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        return None
    target = np.asarray(tuple(cursor), dtype=float)
    distances = np.linalg.norm(values - target[None, :], axis=1)
    index = int(np.argmin(distances))
    return index if distances[index] <= float(radius_px) else None


@dataclass(frozen=True)
class InspectionRow:
    x: float
    y: float
    values: dict[str, Any]

    def tooltip(self, *, digits: int = 6) -> str:
        rendered: list[str] = []
        for key, value in self.values.items():
            if isinstance(value, (float, np.floating)):
                value = format_significant(float(value), digits)
            rendered.append(f"{key}: {value}")
        return "\n".join(rendered)


@dataclass(frozen=True)
class RecordView:
    path: Path
    operation: str
    config: dict[str, Any]
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]
    rows: tuple[InspectionRow, ...]
    columns: tuple[str, ...]


def load_record(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    directory = Path(path).resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "arrays.npz", allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return config, summary, arrays


def _reference_length(config: dict[str, Any]) -> float | None:
    saved = config.get("config", config)
    return saved.get("case", {}).get("actual_lattice_constant_m", saved.get("actual_lattice_constant_m"))


def _path_labels(summary: dict[str, Any], sample_count: int) -> dict[int, str]:
    labels = summary.get("path_labels") or []
    if not labels:
        return {}
    indices = np.rint(np.linspace(0, max(sample_count - 1, 0), len(labels))).astype(int)
    return {int(index): ("Γ" if str(label).lower() in {"gamma", "g", "γ"} else str(label)) for index, label in zip(indices, labels)}


def record_view(path: str | Path, *, frequency_unit: str = "Normalized", component_index: int = 0,
                field_quantity: str = "energy_density") -> RecordView:
    """Build immutable, raw-sample inspection rows for a Studio record."""

    config, summary, arrays = load_record(path)
    operation = OPERATION_ALIASES.get(config.get("identity", {}).get("operation", summary.get("operation", "result")), config.get("identity", {}).get("operation", summary.get("operation", "result")))
    rows: list[InspectionRow] = []

    if operation == "band_structure" and "frequencies" in arrays:
        frequencies = np.asarray(arrays["frequencies"], dtype=float)
        qpoints = np.asarray(arrays.get("qpoints", np.full((len(frequencies), 2), np.nan)), dtype=float)
        factor = frequency_factor(frequency_unit, _reference_length(config))
        labels = _path_labels(summary, len(frequencies))
        for sample in range(len(frequencies)):
            for band in range(frequencies.shape[1]):
                values = {"Band": band + 1, "Sample": sample}
                if sample in labels:
                    values["Path"] = labels[sample]
                values.update({"qₓ": qpoints[sample, 0], "qᵧ": qpoints[sample, 1], frequency_label(frequency_unit): frequencies[sample, band] * factor})
                rows.append(InspectionRow(float(sample), float(frequencies[sample, band] * factor), values))
    elif operation == "berry" and "curvature" in arrays:
        plaquettes = arrays.get("plaquettes")
        points = np.asarray(arrays.get("sample_centers", np.mean(plaquettes, axis=1) if plaquettes is not None else arrays.get("qpoints")), dtype=float)
        curvature = np.asarray(arrays["curvature"], dtype=float).reshape(-1)
        qualifications = summary.get("qualification", {}).get("per_plaquette", summary.get("qualifications", []))
        calculation = config.get("config", {}).get("calculation", {})
        target = (f"Band {calculation.get('band_one_based')}" if calculation.get("berry_target_mode") == "single_band"
                  else f"Bands {calculation.get('berry_first_band')}–{calculation.get('berry_last_band')} composite")
        for index, (point, value) in enumerate(zip(points, curvature)):
            qualified = qualifications[index].get("qualified") if index < len(qualifications) else None
            rows.append(InspectionRow(float(point[0]), float(point[1]), {"Sample": index, "qₓ": point[0], "qᵧ": point[1], "Berry curvature": value, "Target": target, "Qualified": qualified if qualified is not None else "unknown"}))
    elif operation == "efs" and "qpoints" in arrays and "frequencies" in arrays:
        qpoints = np.asarray(arrays["qpoints"], dtype=float)
        frequencies = np.asarray(arrays["frequencies"], dtype=float)
        factor = frequency_factor(frequency_unit, _reference_length(config))
        bands = summary.get("bands_one_based") or [index + 1 for index in range(frequencies.shape[1])]
        for sample, point in enumerate(qpoints):
            for column in range(frequencies.shape[1]):
                value = frequencies[sample, column] * factor
                rows.append(InspectionRow(float(point[0]), float(point[1]), {"Sample": sample, "Band": bands[column] if column < len(bands) else column + 1, "qₓ": point[0], "qᵧ": point[1], frequency_label(frequency_unit): value}))
    elif operation == "fields_energy" and "energy_density" in arrays:
        quantity = np.asarray(arrays.get(field_quantity, arrays["energy_density"]), dtype=float)
        selected = max(0, min(int(component_index), quantity.shape[-1] - 1))
        image = quantity[0, :, :, selected]
        saved_case = config.get("config", {}).get("case", config.get("case"))
        if saved_case:
            basis = model_from_case(saved_case).effective_lattice.direct_basis
            fractional = np.stack(np.meshgrid(np.arange(image.shape[1]) / image.shape[1], np.arange(image.shape[0]) / image.shape[0], indexing="xy"), axis=-1)
            coordinates = fractional @ np.asarray(basis, dtype=float).T
            corners_fractional = np.stack(np.meshgrid(np.arange(image.shape[1] + 1) / image.shape[1], np.arange(image.shape[0] + 1) / image.shape[0], indexing="xy"), axis=-1)
            corners = corners_fractional @ np.asarray(basis, dtype=float).T
            arrays = dict(arrays)
            arrays["display_x_corners"] = corners[:, :, 0]
            arrays["display_y_corners"] = corners[:, :, 1]
        else:
            coordinates = np.stack(np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), indexing="xy"), axis=-1)
        for iy, ix in np.ndindex(image.shape):
            x, y = coordinates[iy, ix]
            rows.append(InspectionRow(float(x), float(y), {"x": x, "y": y, "x index": ix, "y index": iy, "Component": selected, field_quantity: image[iy, ix]}))
    elif operation == "frequency_at_k" and "frequency" in arrays:
        factor = frequency_factor(frequency_unit, _reference_length(config))
        calculation = config.get("config", {}).get("calculation", {})
        qpoint = calculation.get("qpoint", [np.nan, np.nan])
        value = float(np.asarray(arrays["frequency"]).reshape(-1)[0]) * factor
        rows.append(InspectionRow(0.0, value, {"Band": calculation.get("band_one_based", "unknown"), "qₓ": qpoint[0], "qᵧ": qpoint[1], frequency_label(frequency_unit): value}))
    elif operation == "berry_curvature_dipole" and "qpoints" in arrays:
        qpoints = np.asarray(arrays["qpoints"], dtype=float)
        curvature = np.asarray(arrays.get("berry_curvature", np.full(len(qpoints), np.nan))).reshape(-1)
        for index, point in enumerate(qpoints):
            rows.append(InspectionRow(float(point[0]), float(point[1]), {"Sample": index, "qₓ": point[0], "qᵧ": point[1], "Berry curvature": curvature[index]}))

    columns = tuple(dict.fromkeys(key for row in rows for key in row.values))
    return RecordView(Path(path).resolve(), operation, config, summary, arrays, tuple(rows), columns)
