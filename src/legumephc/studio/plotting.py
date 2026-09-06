from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure


def default_plot_style() -> dict[str, Any]:
    return {"width_px": 900, "height_px": 600, "dpi": 100, "title": "", "x_label": "", "y_label": "", "grid": True, "legend": True, "x_limits": None, "y_limits": None, "band_style": "line", "band_line": True, "band_markers": False, "linewidth": 1.5, "marker_size": 4.0, "cmap": "viridis", "berry_coloring": False, "berry_interpolation": False, "berry_vmin": None, "berry_vmax": None, "colorbar": True, "component_index": 0, "field_quantity": "energy_density", "font_size": 10.0}


def _path_label(value: Any) -> str:
    return "Γ" if str(value).strip().lower() in {"gamma", "γ", "g"} else str(value)


def _load_record(record_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    directory = Path(record_path)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return config, summary, arrays


def _status_text(config: dict[str, Any], summary: dict[str, Any]) -> str:
    identity = config.get("identity", {})
    operation = identity.get("operation", summary.get("operation", "result"))
    qualification = summary.get("qualification")
    if isinstance(qualification, dict):
        status = qualification.get("overall_status", qualification.get("status", "unqualified"))
        return f"{operation} | Berry status: {status}"
    if "kind" in summary:
        return f"{operation} | BCD: {summary['kind']}"
    return f"{operation} | {summary.get('status', 'unknown')}"


def _grid_arrays(arrays: dict[str, np.ndarray], summary: dict[str, Any], band_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not {"grid_qpoints", "inside_bz_mask", "frequencies"}.issubset(arrays):
        return None
    shape = tuple(int(value) for value in summary.get("grid_shape", ()))
    if len(shape) != 2 or int(np.prod(shape)) != len(arrays["grid_qpoints"]):
        return None
    frequencies = np.asarray(arrays["frequencies"])
    if band_index < 0 or band_index >= frequencies.shape[1]:
        return None
    grid = np.asarray(arrays["grid_qpoints"]).reshape(*shape, 2)
    mask = np.asarray(arrays["inside_bz_mask"], dtype=bool).reshape(*shape)
    values = np.full(shape, np.nan, dtype=float)
    values.reshape(-1)[mask.reshape(-1)] = frequencies[:, band_index]
    return grid[:, :, 0], grid[:, :, 1], values


def plot_record(record_path: str | Path, style: dict[str, Any] | None = None) -> Figure:
    config, summary, arrays = _load_record(record_path)
    options = {**default_plot_style(), **(style or {})}
    figure = Figure(figsize=(float(options["width_px"]) / float(options["dpi"]), float(options["height_px"]) / float(options["dpi"])), dpi=float(options["dpi"]))
    axis = figure.add_subplot(111)
    operation = config.get("identity", {}).get("operation", summary.get("operation", "result"))
    if operation in {"solve_bands", "band_structure"} and "frequencies" in arrays:
        frequencies = np.asarray(arrays["frequencies"])
        x = np.arange(frequencies.shape[0])
        legacy = options.get("band_style", "line")
        line_enabled = bool(options.get("band_line", legacy != "scatter"))
        markers_enabled = bool(options.get("band_markers", legacy in {"scatter", "cycle"}))
        if not line_enabled and not markers_enabled:
            raise ValueError("enable Band lines, Band markers, or both")
        for band in range(frequencies.shape[1]):
            color = f"C{band % 10}"
            if line_enabled:
                axis.plot(x, frequencies[:, band], color=color, linewidth=float(options.get("linewidth", 1.5)), label=f"Band {band + 1}")
            if markers_enabled:
                axis.scatter(x, frequencies[:, band], color=color, s=float(options.get("marker_size", 4.0)) ** 2, label=f"Band {band + 1}" if not line_enabled else "_nolegend_")
        labels = summary.get("path_labels")
        if labels:
            positions = np.linspace(0, max(0, len(x) - 1), len(labels))
            axis.set_xticks(positions, [_path_label(value) for value in labels])
            if options["x_label"]:
                axis.set_xlabel(options["x_label"])
        else:
            axis.set_xlabel(options["x_label"] or "sample")
        axis.set_ylabel(options["y_label"] or "Normalized frequency (ωa/2πc)")
    elif operation == "frequency_at_k" and "frequency" in arrays:
        axis.bar([0], arrays["frequency"])
        axis.set_ylabel(options["y_label"] or "Normalized frequency (ωa/2πc)")
    elif operation in {"compute_field_observables", "fields_energy"} and "energy_density" in arrays:
        quantity = str(options.get("field_quantity", "energy_density"))
        energy = np.asarray(arrays.get(quantity, arrays["energy_density"]))
        selected = int(options.get("component_index", 0))
        selected = max(0, min(selected, energy.shape[-1] - 1))
        image = energy[0, :, :, selected]
        axis.imshow(image, origin="lower", aspect="equal")
        bands = summary.get("bands_zero_based", [selected])
        band = int(bands[selected]) + 1 if selected < len(bands) else selected + 1
        axis.set_xlabel(options["x_label"] or "unit-cell x")
        axis.set_ylabel(options["y_label"] or "unit-cell y")
    elif operation in {"solve_efs", "efs"} and "qpoints" in arrays and "frequencies" in arrays:
        selected = int(options.get("component_index", 0))
        grid = _grid_arrays(arrays, summary, selected)
        if grid is not None:
            x_grid, y_grid, values_grid = grid
            artist = axis.contourf(x_grid, y_grid, values_grid, levels=12, cmap=options.get("cmap", "viridis"))
            if options.get("colorbar", True):
                figure.colorbar(artist, ax=axis, label="frequency")
        else:
            values = np.asarray(arrays["frequencies"])[:, selected]
            axis.scatter(arrays["qpoints"][:, 0], arrays["qpoints"][:, 1], c=values, label="sparse samples")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(options["x_label"] or "qₓ")
        axis.set_ylabel(options["y_label"] or "qᵧ")
    elif operation in {"solve_berry", "berry"} and "qpoints" in arrays:
        plaquettes = np.asarray(arrays["plaquettes"]) if "plaquettes" in arrays else None
        points = np.mean(plaquettes, axis=1) if plaquettes is not None else np.asarray(arrays["qpoints"])
        values = np.asarray(arrays.get("curvature", np.zeros(len(points)))).reshape(-1)
        if options["berry_coloring"] and len(values) == len(points):
            if options.get("berry_interpolation") and len(points) >= 3:
                artist = axis.tricontourf(points[:, 0], points[:, 1], values, levels=12, cmap=options.get("cmap", "viridis"), vmin=options.get("berry_vmin"), vmax=options.get("berry_vmax"))
            else:
                artist = axis.scatter(points[:, 0], points[:, 1], c=values, cmap=options.get("cmap", "viridis"), vmin=options.get("berry_vmin"), vmax=options.get("berry_vmax"), label="plaquette curvature")
            if options.get("colorbar", True):
                figure.colorbar(artist, ax=axis, label="Berry curvature")
        else:
            axis.scatter(points[:, 0], points[:, 1], label="plaquette centers")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(options["x_label"] or "qₓ")
        axis.set_ylabel(options["y_label"] or "qᵧ")
    elif operation in {"compute_berry_dipole", "berry_curvature_dipole"} and "qpoints" in arrays:
        color = arrays.get("berry_curvature")
        axis.scatter(arrays["qpoints"][:, 0], arrays["qpoints"][:, 1], c=color if color is not None else None, cmap=options.get("cmap", "viridis"), label="samples")
        if "first_moment" in arrays:
            moment = np.asarray(arrays["first_moment"]).reshape(2)
            axis.quiver([0], [0], [moment[0]], [moment[1]], angles="xy", scale_units="xy", scale=1, label="first moment")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(options["x_label"] or "qₓ")
        axis.set_ylabel(options["y_label"] or "qᵧ")
    else:
        axis.text(0.5, 0.5, "No plottable arrays", ha="center", va="center")
    axis.set_title(str(options["title"]) if options["title"] else "")
    if options["grid"]:
        axis.grid(True)
    if options["x_limits"]:
        axis.set_xlim(*options["x_limits"])
    if options["y_limits"]:
        axis.set_ylim(*options["y_limits"])
    if options["legend"] and axis.get_legend_handles_labels()[0]:
        axis.legend()
    axis.tick_params(labelsize=float(options.get("font_size", 10.0)))
    figure.tight_layout()
    return figure


def export_figure(figure: Figure, path: str | Path, *, width_px: int, height_px: int, dpi: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(float(width_px) / dpi, float(height_px) / dpi, forward=True)
    figure.savefig(target, dpi=dpi)
    return target
