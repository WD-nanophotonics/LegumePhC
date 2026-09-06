from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import Affine2D, GeometrySpec, Lattice2D
from ..model import Model2D


def _lattice(case: dict[str, Any]) -> Lattice2D:
    kind = case["lattice"]
    scale = float(case.get("lattice_constant", 1.0))
    if kind == "triangular":
        return Lattice2D.triangular(scale)
    if kind == "square":
        return Lattice2D.square(scale)
    return Lattice2D(np.asarray(case["direct_basis"], dtype=float), kind="custom")


def model_from_case(case: dict[str, Any]) -> Model2D:
    lattice = _lattice(case)
    geometry = case["geometry"]
    motif_dicts = geometry.get("motifs")
    if motif_dicts is None:
        motif_dicts = [geometry]
    kinds = tuple(str(item.get("kind", "circle")) for item in motif_dicts)
    kind = kinds[0]
    sides = tuple(int(item.get("sides", 8)) if item.get("kind", "circle") == "polygon" else 0 for item in motif_dicts)
    spec = GeometrySpec(
        name=str(geometry.get("name", case.get("name", "StudioGeometry"))),
        kind=kind,
        radii=tuple(float(item.get("radius", 0.2)) for item in motif_dicts),
        sides=sides,
        angles_degrees=tuple(float(item.get("angle_degrees", 0.0)) for item in motif_dicts),
        strict_c3=False,
        epsilon_background=float(geometry.get("epsilon_background", 7.29)),
        epsilon_inclusion=float(geometry.get("epsilon_inclusion", 1.0)),
        motif_kinds=kinds,
        motif_epsilons=tuple(float(item.get("epsilon", geometry.get("epsilon_inclusion", 1.0))) for item in motif_dicts),
        direct_basis=lattice.direct_basis,
        centers=np.asarray([item.get("center", [0.5, 0.5]) for item in motif_dicts], dtype=float),
    )
    affine_raw = case.get("affine", {})
    affine = Affine2D(np.asarray(affine_raw["linear"], dtype=float), np.asarray(affine_raw["translation"], dtype=float))
    return Model2D(spec, lattice, affine=affine, basis_policy=str(case.get("basis_policy", "auto")))


def zero_based_band(value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError("user band labels are one-based and must be at least 1")
    return value - 1


def zero_based_bands(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(zero_based_band(value) for value in values)
