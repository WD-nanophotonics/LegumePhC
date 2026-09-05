from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Affine2D, GeometrySpec, Lattice2D, geometry_spec, m7_orbit


@dataclass(frozen=True)
class Model2D:
    """Stable domain description for a two-dimensional PWE study."""

    geometry: GeometrySpec
    lattice: Lattice2D
    affine: Affine2D = field(default_factory=Affine2D)
    basis_policy: str = "auto"
    point_group: str | None = None
    closure_qpoints: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.basis_policy not in {"auto", "native", "circular", "closed"}:
            raise ValueError("basis_policy must be auto, native, circular, or closed")
        if self.point_group not in {None, "C3", "C4"}:
            raise ValueError("point_group must be None, C3, or C4")

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
        point_group = "C3" if lattice.kind == "triangular" and spec.strict_c3 and affine_is_identity else None
        closure = m7_orbit(config) if point_group == "C3" else None
        return cls(spec, lattice, affine, basis_policy, point_group, closure)

    @property
    def identity(self) -> dict[str, object]:
        return {
            "model": "Model2D",
            "geometry": self.geometry.name,
            "affine": {"linear": self.affine.linear.tolist(), "translation": self.affine.translation.tolist()},
            "basis": {
                "policy": self.basis_policy,
                "direct_basis": self.lattice.direct_basis.tolist(),
                "reciprocal_basis": self.lattice.reciprocal_basis.tolist(),
            },
            "point_group": self.point_group,
            "lattice": self.lattice.kind,
        }
