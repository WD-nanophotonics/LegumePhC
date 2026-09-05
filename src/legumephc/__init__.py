"""Windows-native Legume-backed two-dimensional PWE domain core."""

from .config import BenchmarkConfig, load_benchmark
from .geometry import Affine2D, GeometrySpec, Lattice2D, geometry_spec, m7_orbit
from .model import Model2D
from .solver import frequency_at_k, solve_bands

__all__ = [
    "Affine2D", "BenchmarkConfig", "GeometrySpec", "Lattice2D", "Model2D",
    "frequency_at_k", "geometry_spec", "load_benchmark", "m7_orbit", "solve_bands",
]
