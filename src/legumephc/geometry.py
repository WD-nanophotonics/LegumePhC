from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import BenchmarkConfig


DIRECT_BASIS = np.array([[0.5, 0.5], [math.sqrt(3) / 2, -math.sqrt(3) / 2]], dtype=float)
SUBLATTICE_CENTERS = np.array([[0.0, 0.0], [0.5, 1.0 / (2 * math.sqrt(3))]], dtype=float)


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    kind: str
    radii: tuple[float, float]
    sides: tuple[int, int] | None
    angles_degrees: tuple[float, float]
    strict_c3: bool
    epsilon_background: float
    epsilon_inclusion: float


def geometry_spec(config: BenchmarkConfig, name: str) -> GeometrySpec:
    try:
        raw = config.raw["geometries"][name]
    except KeyError as exc:
        raise ValueError(f"unknown geometry {name!r}") from exc
    a = config.lattice_constant_nm
    sides = tuple(map(int, raw["sides"])) if raw["kind"] == "polygon" else None
    angles = tuple(map(float, raw.get("angles_degrees", (0.0, 0.0))))
    return GeometrySpec(
        name=name,
        kind=str(raw["kind"]),
        radii=tuple(float(value) / a for value in raw["radii_nm"]),
        sides=sides,
        angles_degrees=angles,
        strict_c3=bool(raw["strict_c3"]),
        epsilon_background=config.background_epsilon,
        epsilon_inclusion=config.inclusion_epsilon,
    )


def polygon_vertices(radius: float, sides: int, angle_degrees: float, center: np.ndarray) -> np.ndarray:
    phase = 2 * math.pi * np.arange(sides) / sides + math.radians(angle_degrees)
    local = np.column_stack((radius * np.sin(phase), radius * np.cos(phase)))
    # MePhC's historical sin/cos convention enumerates clockwise.  Legume
    # requires counter-clockwise edges; reversing preserves the exact polygon.
    return (local + np.asarray(center, dtype=float))[::-1]


def rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def m7_orbit(config: BenchmarkConfig) -> np.ndarray:
    m7 = config.raw["m7"]
    center = np.asarray(m7["k_center_reciprocal"], dtype=float)
    seed = center - np.asarray(m7["seed_offset_reciprocal"], dtype=float)
    return np.asarray([center + rotation(angle) @ (seed - center) for angle in m7["rotation_degrees"]])


def area_matched_radius(radius: float, source_sides: int, target_sides: int) -> float:
    return radius * math.sqrt(
        source_sides * math.sin(2 * math.pi / source_sides)
        / (target_sides * math.sin(2 * math.pi / target_sides))
    )


def strict_c3_by_construction(spec: GeometrySpec) -> bool:
    if spec.kind == "circle":
        return True
    assert spec.sides is not None
    return all(sides % 3 == 0 for sides in spec.sides)


def c3_geometry_residual(spec: GeometrySpec) -> float:
    """Return a solver-free Hausdorff residual for the periodic motif under C3."""

    transform = rotation(120.0)
    inverse_basis = np.linalg.inv(DIRECT_BASIS)
    residual = 0.0
    for index, center in enumerate(SUBLATTICE_CENTERS):
        displacement = transform @ center - center
        lattice_coefficients = inverse_basis @ displacement
        residual = max(residual, float(np.max(np.abs(lattice_coefficients - np.rint(lattice_coefficients)))))
        if spec.kind == "circle":
            continue
        assert spec.sides is not None
        local = polygon_vertices(spec.radii[index], spec.sides[index], spec.angles_degrees[index], np.zeros(2))
        rotated = local @ transform.T
        pairwise = np.linalg.norm(rotated[:, None, :] - local[None, :, :], axis=2)
        residual = max(residual, float(max(np.max(np.min(pairwise, axis=1)), np.max(np.min(pairwise, axis=0)))))
    return residual
