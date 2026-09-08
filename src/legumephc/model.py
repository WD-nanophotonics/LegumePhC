from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from .geometry import Affine2D, GeometrySpec, Lattice2D, geometry_spec, m7_orbit, point_group_geometry_residual, polygon_vertices


@dataclass(frozen=True)
class Model2D:
    """Stable domain description for a two-dimensional PWE study."""

    geometry: GeometrySpec
    lattice: Lattice2D
    affine: Affine2D = field(default_factory=Affine2D)
    basis_policy: str = "auto"
    point_group: str | None = field(default=None, init=False)
    unverified_point_group: str | None = None
    closure_qpoints: np.ndarray | None = None
    actual_lattice_constant_m: float | None = None

    def __post_init__(self) -> None:
        if self.actual_lattice_constant_m is not None:
            from .units import reference_length
            reference_length(self.actual_lattice_constant_m, "m")
        if self.basis_policy not in {"auto", "native", "circular", "closed"}:
            raise ValueError("basis_policy must be auto, native, circular, or closed")
        if self.unverified_point_group not in {None, "C3", "C4"}:
            raise ValueError("unverified_point_group must be None, C3, or C4")
        affine_is_identity = np.allclose(self.affine.linear, np.eye(2)) and np.allclose(self.affine.translation, 0.0)
        verified = None
        for candidate, lattice_kind in (("C3", "triangular"), ("C4", "square")):
            if affine_is_identity and self.lattice.kind == lattice_kind:
                if point_group_geometry_residual(self.geometry, self.lattice, candidate) <= 1e-10:
                    verified = candidate
                    break
        object.__setattr__(self, "point_group", verified)

    @classmethod
    def from_benchmark(
        cls,
        config,
        name: str,
        *,
        lattice: Lattice2D | None = None,
        affine: Affine2D | None = None,
        basis_policy: str = "auto",
    ) -> "Model2D":
        spec = geometry_spec(config, name)
        lattice = Lattice2D.triangular() if lattice is None else lattice
        affine = Affine2D() if affine is None else affine
        affine_is_identity = np.allclose(affine.linear, np.eye(2)) and np.allclose(affine.translation, 0.0)
        closure = m7_orbit(config) if lattice.kind == "triangular" and spec.strict_c3 and affine_is_identity else None
        return cls(spec, lattice, affine, basis_policy=basis_policy, closure_qpoints=closure)

    @property
    def identity(self) -> dict[str, object]:
        return {
            "model": "Model2D",
            "geometry": self.geometry.name,
            "motifs": {
                "kinds": list(self.geometry.motif_kinds),
                "radii": list(self.geometry.radii),
                "centers": np.asarray(self.geometry.centers).tolist(),
                "angles_degrees": list(self.geometry.angles_degrees),
                "sides": None if self.geometry.sides is None else list(self.geometry.sides),
                "epsilons": list(self.geometry.motif_epsilons),
            },
            "affine": {"linear": self.affine.linear.tolist(), "translation": self.affine.translation.tolist()},
            "basis": {
                "policy": self.basis_policy,
                "direct_basis": self.lattice.direct_basis.tolist(),
                "reciprocal_basis": self.lattice.reciprocal_basis.tolist(),
            },
            "point_group": self.point_group,
            "unverified_point_group": self.unverified_point_group,
            "lattice": self.lattice.kind,
        }

    @property
    def effective_lattice(self) -> Lattice2D:
        """Return the direct lattice after the model's affine transform."""

        identity = np.allclose(self.affine.linear, np.eye(2))
        kind = self.lattice.kind if identity else "custom"
        return Lattice2D(self.affine.linear @ self.lattice.direct_basis, kind=kind)

    @property
    def effective_geometry(self) -> GeometrySpec:
        """Return the motif after the same affine map used for the lattice."""

        if np.allclose(self.affine.linear, np.eye(2)) and np.allclose(self.affine.translation, 0.0):
            return self.geometry
        ellipse_parameters: list[tuple[float, float, float] | None] = []
        transformed_vertices: list[np.ndarray | None] = []
        singular_vectors, singular_values, _ = np.linalg.svd(self.affine.linear)
        phi = float(np.arctan2(singular_vectors[1, 0], singular_vectors[0, 0]))
        for index, center in enumerate(self.geometry.centers):
            if self.geometry.motif_kinds[index] == "circle":
                radius = self.geometry.radii[index]
                ellipse_parameters.append((float(radius * singular_values[0]), float(radius * singular_values[1]), phi))
                transformed_vertices.append(None)
            else:
                assert self.geometry.sides is not None
                source = polygon_vertices(self.geometry.radii[index], self.geometry.sides[index], self.geometry.angles_degrees[index], center)
                transformed_vertices.append(self.affine.apply(source))
                ellipse_parameters.append(None)
        return GeometrySpec(
            name=self.geometry.name,
            kind=self.geometry.kind,
            radii=self.geometry.radii,
            sides=self.geometry.sides,
            angles_degrees=self.geometry.angles_degrees,
            strict_c3=False,
            epsilon_background=self.geometry.epsilon_background,
            epsilon_inclusion=self.geometry.epsilon_inclusion,
            motif_kinds=self.geometry.motif_kinds,
            motif_epsilons=self.geometry.motif_epsilons,
            direct_basis=self.effective_lattice.direct_basis,
            centers=self.affine.apply(self.geometry.centers),
            ellipse_parameters=tuple(ellipse_parameters),
            transformed_vertices=tuple(transformed_vertices) if transformed_vertices else None,
        )

    @property
    def cache_identity(self) -> str:
        canonical = json.dumps(self.identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
