from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scipy.spatial import Voronoi

from ..geometry import first_bz_vertices
from .profile import model_from_case
from ..units import frequency_factor, frequency_label


def _clip_half_plane(polygon: np.ndarray, normal: np.ndarray, limit: float, *, tolerance: float = 1e-12) -> np.ndarray:
    output: list[np.ndarray] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        start_value = float(np.dot(normal, start) - limit)
        end_value = float(np.dot(normal, end) - limit)
        start_inside = start_value <= tolerance
        end_inside = end_value <= tolerance
        if start_inside:
            output.append(start)
        if start_inside != end_inside:
            fraction = start_value / (start_value - end_value)
            output.append(start + fraction * (end - start))
    return np.asarray(output, dtype=float)


def sample_cell_polygons(points: np.ndarray, domain_outline: np.ndarray) -> list[np.ndarray]:
    """Return Voronoi ownership cells clipped to a convex sampling domain."""

    points = np.asarray(points, dtype=float)
    raw_domain = np.asarray(domain_outline, dtype=float)
    domain_values: list[np.ndarray] = []
    for vertex in raw_domain:
        if not domain_values or np.linalg.norm(vertex - domain_values[-1]) > 1e-12:
            domain_values.append(vertex)
    if len(domain_values) > 1 and np.linalg.norm(domain_values[0] - domain_values[-1]) <= 1e-12:
        domain_values.pop()
    domain = np.asarray(domain_values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("sample centres must have shape (N, 2)")
    if domain.ndim != 2 or domain.shape[1] != 2 or len(domain) < 3:
        raise ValueError("domain outline must be a polygon")
    if len(np.unique(np.round(points, 13), axis=0)) != len(points):
        raise ValueError("sample centres must be unique")
    neighbours = [set(range(len(points))) - {index} for index in range(len(points))]
    if len(points) >= 4:
        try:
            voronoi = Voronoi(points)
            neighbours = [set() for _ in points]
            for first, second in voronoi.ridge_points:
                neighbours[int(first)].add(int(second))
                neighbours[int(second)].add(int(first))
        except Exception:
            pass
    cells: list[np.ndarray] = []
    for index, point in enumerate(points):
        cell = domain.copy()
        for other_index in neighbours[index]:
            other = points[other_index]
            normal = other - point
            limit = (float(np.dot(other, other)) - float(np.dot(point, point))) / 2.0
            cell = _clip_half_plane(cell, normal, limit)
            if len(cell) == 0:
                break
        cells.append(cell)
    return cells


def default_plot_style() -> dict[str, Any]:
    return {"width_px": 900, "height_px": 600, "dpi": 100, "title": "", "x_label": "", "y_label": "", "grid": True, "legend": True, "x_limits": None, "y_limits": None, "band_style": "line", "band_line": True, "band_markers": False, "linewidth": 1.5, "marker_size": 4.0, "cmap": "RdBu_r", "berry_coloring": True, "berry_render_mode": "sample_cells", "show_sample_centers": False, "berry_interpolation": False, "berry_vmin": None, "berry_vmax": None, "colorbar": True, "component_index": 0, "field_quantity": "energy_density", "font_size": 10.0, "frequency_unit": "Normalized"}


def _berry_norm(values: np.ndarray, options: dict[str, Any]) -> Normalize:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    extent = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1e-15)
    vmin = options.get("berry_vmin")
    vmax = options.get("berry_vmax")
    vmin = -extent if vmin is None else float(vmin)
    vmax = extent if vmax is None else float(vmax)
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("Berry color minimum must be smaller than maximum")
    return Normalize(vmin=vmin, vmax=vmax)


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
    unit = options.get("frequency_unit", "Normalized")
    factor = 1.0
    if operation in {"solve_bands", "band_structure", "frequency_at_k", "solve_efs", "efs"}:
        saved = config.get("config", {})
        length = saved.get("case", {}).get("actual_lattice_constant_m", saved.get("actual_lattice_constant_m"))
        factor = frequency_factor(unit, length)
        arrays = dict(arrays)
        for key in ("frequencies", "frequency"):
            if key in arrays:
                arrays[key] = np.asarray(arrays[key]) * factor
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
        axis.set_ylabel(options["y_label"] or frequency_label(unit))
    elif operation == "frequency_at_k" and "frequency" in arrays:
        axis.bar([0], arrays["frequency"])
        axis.set_ylabel(options["y_label"] or frequency_label(unit))
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
            levels = options.get("efs_levels")
            if levels is not None:
                levels = np.asarray(levels, dtype=float)
                if len(levels) < 1 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
                    raise ValueError("EFS levels must be finite and strictly increasing")
                artist = axis.contour(x_grid, y_grid, values_grid, levels=levels, cmap=options.get("cmap", "viridis"))
            else:
                artist = axis.contourf(x_grid, y_grid, values_grid, levels=12, cmap=options.get("cmap", "viridis"))
            if options.get("colorbar", True):
                figure.colorbar(artist, ax=axis, label=frequency_label(unit))
        else:
            values = np.asarray(arrays["frequencies"])[:, selected]
            axis.scatter(arrays["qpoints"][:, 0], arrays["qpoints"][:, 1], c=values, label="sparse samples")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(options["x_label"] or "qₓ")
        axis.set_ylabel(options["y_label"] or "qᵧ")
    elif operation in {"solve_berry", "berry"} and "qpoints" in arrays:
        plaquettes = np.asarray(arrays["plaquettes"]) if "plaquettes" in arrays else None
        points = np.asarray(arrays["sample_centers"]) if "sample_centers" in arrays else (np.mean(plaquettes, axis=1) if plaquettes is not None else np.asarray(arrays["qpoints"]))
        values = np.asarray(arrays.get("curvature", np.zeros(len(points)))).reshape(-1)
        if len(values) == len(points):
            norm = _berry_norm(values, options)
            render_mode = options.get("berry_render_mode", "sample_cells")
            if (render_mode == "linear_interpolation" or options.get("berry_interpolation")) and len(points) >= 3:
                from scipy.interpolate import griddata
                from matplotlib.path import Path as MplPath
                resolution = max(8, int(options.get("interpolation_resolution", 160)))
                x = np.linspace(float(np.min(points[:, 0])), float(np.max(points[:, 0])), resolution)
                y = np.linspace(float(np.min(points[:, 1])), float(np.max(points[:, 1])), resolution)
                xx, yy = np.meshgrid(x, y)
                zz = griddata(points, values, (xx, yy), method="linear")
                if "domain_outline" in arrays:
                    inside = MplPath(np.asarray(arrays["domain_outline"])).contains_points(np.column_stack((xx.ravel(), yy.ravel())), radius=1e-12).reshape(xx.shape)
                    zz = np.where(inside, zz, np.nan)
                artist = axis.contourf(xx, yy, zz, levels=24, cmap=options.get("cmap", "RdBu_r"), norm=norm)
            elif len(points) >= 2:
                domain = np.asarray(arrays["domain_outline"]) if "domain_outline" in arrays else None
                legacy_sampling = "sample_centers" not in arrays
                if domain is None:
                    try:
                        model = model_from_case(config.get("config", {}).get("case", config.get("case", {})))
                        domain = first_bz_vertices(model.effective_lattice)
                    except (KeyError, TypeError, ValueError):
                        margin = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-3) * 0.05
                        lower, upper = np.min(points, axis=0) - margin, np.max(points, axis=0) + margin
                        domain = np.asarray([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
                cells = sample_cell_polygons(points, domain)
                artist = PolyCollection(cells, array=values, cmap=options.get("cmap", "RdBu_r"), norm=norm, edgecolors="none", label="legacy Cartesian sampling" if legacy_sampling else "sample-cell tiling")
                axis.add_collection(artist)
                axis.autoscale_view()
            else:
                artist = axis.scatter(points[:, 0], points[:, 1], c=values, cmap=options.get("cmap", "RdBu_r"), norm=norm)
            if options.get("colorbar", True):
                figure.colorbar(artist, ax=axis, label="Berry curvature")
            per_plaquette = summary.get("qualification", {}).get("per_plaquette", [])
            invalid = np.asarray([not bool(item.get("qualified", False)) for item in per_plaquette], dtype=bool)
            if len(invalid) == len(points) and np.any(invalid):
                axis.scatter(points[invalid, 0], points[invalid, 1], marker="x", s=45, linewidths=1.5, color="black", label="unqualified")
            if options.get("show_sample_centers", False):
                axis.scatter(points[:, 0], points[:, 1], s=8, facecolors="none", edgecolors="black", linewidths=0.5, label="sample centres")
            try:
                model = model_from_case(config.get("config", {}).get("case", config.get("case", {})))
                bz = first_bz_vertices(model.effective_lattice)
                closed = np.vstack((bz, bz[0]))
                axis.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.0, label="first BZ")
            except (KeyError, TypeError, ValueError):
                pass
        else:
            raise ValueError("Berry record has mismatched plaquette centres and curvature values")
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
