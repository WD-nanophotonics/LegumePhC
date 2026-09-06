from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import polygon_vertices
from .profile import model_from_case


def _polygon_mask(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = points[:, None, :] - vertices[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1e-12, axis=1)


def _motif_mask(points: np.ndarray, spec, basis: np.ndarray) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for index, center in enumerate(spec.centers):
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                shift = np.asarray(basis) @ np.array([n1, n2], dtype=float)
                local = points - center - shift
                if spec.motif_kinds[index] == "circle":
                    ellipse = spec.ellipse_parameters[index] if spec.ellipse_parameters else None
                    if ellipse is None:
                        result |= np.sum(local * local, axis=1) <= spec.radii[index] ** 2
                    else:
                        rx, ry, phi = ellipse
                        x1 = local[:, 0] * np.cos(phi) + local[:, 1] * np.sin(phi)
                        y1 = -local[:, 0] * np.sin(phi) + local[:, 1] * np.cos(phi)
                        result |= (x1 / rx) ** 2 + (y1 / ry) ** 2 <= 1.0
                else:
                    assert spec.sides is not None
                    vertices = spec.transformed_vertices[index] if spec.transformed_vertices is not None and spec.transformed_vertices[index] is not None else polygon_vertices(spec.radii[index], spec.sides[index], spec.angles_degrees[index], center)
                    result |= _polygon_mask(points - shift, vertices)
    return result


def preview_geometry(case: dict[str, Any], *, view: str = "unit_cell", size: int = 128) -> dict[str, Any]:
    """Build a solver-free geometry/epsilon preview for the Studio canvas."""

    if view not in {"unit_cell", "motif_array", "lattice_sites", "epsilon"}:
        raise ValueError("view must be unit_cell, motif_array, lattice_sites, or epsilon")
    if int(size) < 8:
        raise ValueError("preview size must be at least 8")
    model = model_from_case(case)
    lattice = model.effective_lattice
    spec = model.effective_geometry
    if view == "lattice_sites":
        sites = np.asarray([n1 * lattice.direct_basis[:, 0] + n2 * lattice.direct_basis[:, 1] for n1 in range(-1, 2) for n2 in range(-1, 2)], dtype=float)
        return {"view": view, "points": sites, "epsilon": None, "shape": None, "equal_aspect": True, "direct_basis": lattice.direct_basis.copy()}
    extent = (0.0, 1.0) if view in {"unit_cell", "epsilon"} else (-1.0, 2.0)
    coordinates = np.linspace(extent[0], extent[1], int(size))
    fractional = np.stack(np.meshgrid(coordinates, coordinates, indexing="xy"), axis=-1)
    points = (fractional @ lattice.direct_basis.T).reshape(-1, 2)
    mask = _motif_mask(points, spec, lattice.direct_basis).reshape(size, size)
    epsilon_flat = np.full(len(points), spec.epsilon_background, dtype=float)
    for index in range(len(spec.radii)):
        motif_case = np.zeros(len(points), dtype=bool)
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                shift = np.asarray(lattice.direct_basis) @ np.array([n1, n2], dtype=float)
                # Reuse the single-motif helper by constructing the same local
                # mask explicitly; this keeps preview independent of Legume.
                center = spec.centers[index]
                local = points - center - shift
                if spec.motif_kinds[index] == "circle":
                    motif_case |= np.sum(local * local, axis=1) <= spec.radii[index] ** 2
                else:
                    vertices = spec.transformed_vertices[index] if spec.transformed_vertices is not None and spec.transformed_vertices[index] is not None else polygon_vertices(spec.radii[index], spec.sides[index], spec.angles_degrees[index], center)
                    motif_case |= _polygon_mask(points - shift, vertices)
        epsilon_flat[motif_case] = spec.motif_epsilons[index]
    epsilon = epsilon_flat.reshape(size, size)
    return {
        "view": view,
        "points": points.reshape(size, size, 2),
        "epsilon": epsilon.reshape(size, size),
        "motif_mask": mask,
        "shape": (int(size), int(size)),
        "equal_aspect": True,
        "direct_basis": lattice.direct_basis.copy(),
        "point_group": model.point_group,
    }
