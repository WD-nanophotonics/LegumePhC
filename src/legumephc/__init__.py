"""Windows-native Legume-backed two-dimensional PWE domain core."""

from .config import BenchmarkConfig, load_benchmark
from .berry import first_bz_plaquettes, solve_berry
from .dipole import compute_berry_dipole
from .geometry import Affine2D, GeometrySpec, Lattice2D, first_bz_vertices, first_bz_vertices_physical, geometry_spec, m7_orbit, square_circle_spec
from .model import Model2D
from .observables import compute_field_observables
from .solver import frequency_at_k, solve_bands
from .efs import solve_efs

__all__ = [
    "Affine2D", "BenchmarkConfig", "GeometrySpec", "Lattice2D", "Model2D",
    "compute_berry_dipole", "compute_field_observables", "frequency_at_k", "geometry_spec",
    "first_bz_vertices", "first_bz_vertices_physical", "first_bz_plaquettes", "load_benchmark", "m7_orbit", "solve_bands", "solve_berry", "solve_efs", "square_circle_spec",
]
