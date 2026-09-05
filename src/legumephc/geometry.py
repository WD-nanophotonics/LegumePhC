from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .config import BenchmarkConfig


DIRECT_BASIS = np.array([[0.5, 0.5], [math.sqrt(3) / 2, -math.sqrt(3) / 2]], dtype=float)
SUBLATTICE_CENTERS = np.array([[0.0, 0.0], [0.5, 1.0 / (2 * math.sqrt(3))]], dtype=float)


@dataclass(frozen=True)
class Lattice2D:
    """A non-singular two-dimensional direct lattice."""

    direct_basis: np.ndarray
    kind: str = "custom"

    def __post_init__(self) -> None:
        basis = np.asarray(self.direct_basis, dtype=float)
        if basis.shape != (2, 2) or abs(float(np.linalg.det(basis))) < 1e-14:
            raise ValueError("direct_basis must be a non-singular 2x2 matrix")
        object.__setattr__(self, "direct_basis", basis)

    @property
    def reciprocal_basis_reduced(self) -> np.ndarray:
        return np.linalg.inv(self.direct_basis).T

    @property
    def reciprocal_basis(self) -> np.ndarray:
        return 2.0 * math.pi * self.reciprocal_basis_reduced

    @classmethod
    def triangular(cls, lattice_constant: float = 1.0) -> "Lattice2D":
        return cls(lattice_constant * DIRECT_BASIS, kind="triangular")

    @classmethod
    def square(cls, lattice_constant: float = 1.0) -> "Lattice2D":
        return cls(lattice_constant * np.eye(2), kind="square")


@dataclass(frozen=True)
class Affine2D:
    linear: np.ndarray = field(default_factory=lambda: np.eye(2))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        linear = np.asarray(self.linear, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if linear.shape != (2, 2) or translation.shape != (2,):
            raise ValueError("affine transform must contain a 2x2 linear part and 2-vector translation")
        if abs(float(np.linalg.det(linear))) < 1e-14:
            raise ValueError("affine linear part must be non-singular")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.linear.T + self.translation


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    kind: str
    radii: tuple[float, ...]
    sides: tuple[int, ...] | None
    angles_degrees: tuple[float, ...]
    strict_c3: bool
    epsilon_background: float
    epsilon_inclusion: float
    direct_basis: np.ndarray = field(default_factory=lambda: DIRECT_BASIS.copy())
    centers: np.ndarray = field(default_factory=lambda: SUBLATTICE_CENTERS.copy())

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) != len(self.radii):
            raise ValueError("centers must contain one 2-vector for every motif radius")
        if self.kind == "polygon" and (self.sides is None or len(self.sides) != len(self.radii)):
            raise ValueError("polygon geometry requires one side count per motif radius")
        if len(self.angles_degrees) != len(self.radii):
            raise ValueError("angles_degrees must contain one angle per motif radius")
        object.__setattr__(self, "centers", centers)


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
        direct_basis=DIRECT_BASIS.copy(),
        centers=SUBLATTICE_CENTERS.copy(),
    )


def square_circle_spec(
    *,
    radius: float = 0.2,
    epsilon_background: float = 7.29,
    epsilon_inclusion: float = 1.0,
    name: str = "SquareCircle",
) -> GeometrySpec:
    """Create the verified one-circle square-lattice C4 control geometry."""

    return GeometrySpec(
        name=name,
        kind="circle",
        radii=(float(radius),),
        sides=None,
        angles_degrees=(0.0,),
        strict_c3=False,
        epsilon_background=float(epsilon_background),
        epsilon_inclusion=float(epsilon_inclusion),
        direct_basis=np.eye(2),
        centers=np.array([[0.5, 0.5]], dtype=float),
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


def point_group_operations(point_group: str) -> tuple[np.ndarray, ...]:
    """Return the verified linear operations for the supported cyclic groups."""

    orders = {"C3": 3, "C4": 4}
    try:
        order = orders[point_group]
    except KeyError as exc:
        raise ValueError("point_group must be C3 or C4") from exc
    return tuple(rotation(360.0 * index / order) for index in range(order))


def point_group_geometry_residual(spec: GeometrySpec, lattice: Lattice2D, point_group: str) -> float:
    """Measure motif and lattice compatibility with a finite cyclic group."""

    operations = point_group_operations(point_group)
    direct_basis = np.asarray(lattice.direct_basis, dtype=float)
    inverse_basis = np.linalg.inv(direct_basis)
    residual = 0.0
    for operation in operations[1:]:
        operation_in_basis = inverse_basis @ operation @ direct_basis
        residual = max(residual, float(np.max(np.abs(operation_in_basis - np.rint(operation_in_basis)))))
        for index, center in enumerate(spec.centers):
            transformed_center = operation @ center
            displacements = transformed_center[None, :] - spec.centers
            coefficients = displacements @ inverse_basis.T
            periodic = coefficients - np.rint(coefficients)
            candidate = int(np.argmin(np.linalg.norm(periodic, axis=1)))
            residual = max(residual, float(np.linalg.norm(periodic[candidate])))
            if np.linalg.norm(periodic[candidate]) > 1e-10:
                continue
            if abs(spec.radii[index] - spec.radii[candidate]) > residual:
                residual = abs(spec.radii[index] - spec.radii[candidate])
            if spec.kind == "polygon":
                assert spec.sides is not None
                if spec.sides[index] != spec.sides[candidate]:
                    residual = max(residual, 1.0)
                    continue
                source = polygon_vertices(spec.radii[index], spec.sides[index], spec.angles_degrees[index], np.zeros(2))
                target = polygon_vertices(spec.radii[candidate], spec.sides[candidate], spec.angles_degrees[candidate], np.zeros(2))
                rotated = source @ operation.T
                pairwise = np.linalg.norm(rotated[:, None, :] - target[None, :, :], axis=2)
                residual = max(residual, float(max(np.max(np.min(pairwise, axis=1)), np.max(np.min(pairwise, axis=0)))))
    return residual


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


def first_bz_labels(lattice: Lattice2D) -> tuple[str, ...]:
    if lattice.kind == "triangular":
        return ("Gamma", "K", "M", "Gamma")
    if lattice.kind == "square":
        return ("Gamma", "X", "M", "Gamma")
    return ("Gamma", "P1", "P2", "Gamma")


def first_bz_vertices(lattice: Lattice2D) -> np.ndarray:
    """Return the actual Wigner-Seitz vertices around reciprocal-space Gamma."""

    from scipy.spatial import Voronoi

    basis = lattice.reciprocal_basis
    integer_points = np.asarray(
        [[n1, n2] for n1 in range(-3, 4) for n2 in range(-3, 4) if (n1, n2) != (0, 0)],
        dtype=float,
    )
    points = np.vstack((np.zeros((1, 2)), (basis @ integer_points.T).T))
    diagram = Voronoi(points)
    region = diagram.regions[diagram.point_region[0]]
    if not region or -1 in region:
        raise ValueError("could not construct a bounded first Brillouin zone")
    vertices = diagram.vertices[np.asarray(region, dtype=int)]
    center = np.mean(vertices, axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def identity_path(lattice: Lattice2D, samples_per_segment: int = 16) -> tuple[tuple[str, ...], np.ndarray]:
    """Return a high-symmetry identity path with honest generic labels."""

    reciprocal = lattice.reciprocal_basis_reduced
    if lattice.kind == "triangular":
        reduced = np.array([[0.0, 0.0], [2.0 / 3.0, 0.0], [0.5, 0.5], [0.0, 0.0]])
    elif lattice.kind == "square":
        reduced = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]])
        vertices = (lattice.reciprocal_basis_reduced @ reduced.T).T
    else:
        bz = first_bz_vertices(lattice)
        vertices = np.vstack((np.zeros(2), bz[0], bz[1], np.zeros(2)))
    if lattice.kind == "triangular":
        vertices = (reciprocal @ reduced.T).T
    points: list[np.ndarray] = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        points.extend(np.linspace(start, end, samples_per_segment, endpoint=False))
    points.append(vertices[-1])
    return first_bz_labels(lattice), np.asarray(points)


def c3_geometry_residual(spec: GeometrySpec) -> float:
    """Return a solver-free Hausdorff residual for the periodic motif under C3."""

    return point_group_geometry_residual(spec, Lattice2D(DIRECT_BASIS, kind="triangular"), "C3")
