"""Typed, shared parameters for direct PyCharm study entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MaterialParameters:
    background_epsilon: float = 7.29
    inclusion_epsilon: float = 1.0


@dataclass(frozen=True)
class MotifParameters:
    name: str
    kind: Literal["circle", "polygon"] = "circle"
    radius: float = 0.2
    sides: int | None = None
    angle_degrees: float = 0.0
    center: tuple[float, float] = (0.5, 0.5)
    epsilon: float | None = None


@dataclass(frozen=True)
class AffineParameters:
    linear: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    translation: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class CaseParameters:
    lattice: Literal["triangular", "square", "custom"]
    material: MaterialParameters
    motifs: tuple[MotifParameters, ...]
    affine: AffineParameters = field(default_factory=AffineParameters)
    lattice_constant: float = 1.0
    direct_basis: tuple[tuple[float, float], tuple[float, float]] | None = None
    basis_policy: Literal["auto", "native", "circular", "closed"] = "auto"
    name: str = ""


@dataclass(frozen=True)
class SolverParameters:
    gmax: float = 2.0
    numeig: int = 4
    polarization: Literal["te", "tm"] = "te"


@dataclass(frozen=True)
class FrequencyParameters:
    solver: SolverParameters = field(default_factory=SolverParameters)
    qpoint: tuple[float, float] = (0.2, 0.07)
    band_one_based: int = 2


@dataclass(frozen=True)
class BandParameters:
    solver: SolverParameters = field(default_factory=SolverParameters)
    path: str = "identity"
    samples_per_segment: int = 16


@dataclass(frozen=True)
class FieldsParameters:
    solver: SolverParameters = field(default_factory=SolverParameters)
    qpoint: tuple[float, float] = (0.2, 0.07)
    bands_one_based: tuple[int, ...] = (2, 3)
    grid_size: int = 8


@dataclass(frozen=True)
class EFSParameters:
    solver: SolverParameters = field(default_factory=SolverParameters)
    bands_one_based: tuple[int, ...] = (2, 3)
    grid_size: int = 5


@dataclass(frozen=True)
class BerryParameters:
    solver: SolverParameters = field(default_factory=SolverParameters)
    sampling_mode: Literal["single_plaquette", "first_bz_grid", "explicit_centers"] = "first_bz_grid"
    center: tuple[float, float] = (0.2, 0.07)
    step: float = 0.02
    grid_size: int = 3
    centers: tuple[tuple[float, float], ...] = ()
    bands_one_based: tuple[int, ...] = (2, 3)
    rank: int = 2
    convergence_status: Literal["NOT_ASSESSED", "CONVERGED", "FAILED"] = "NOT_ASSESSED"


@dataclass(frozen=True)
class BCDParameters:
    berry_record_path: str | None = None
    frequency_window: tuple[float, float] | None = None
    frequency_samples: tuple[float, ...] | None = None
    response_weights: tuple[float, ...] | None = None
    occupation: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PlotParameters:
    width_px: int = 900
    height_px: int = 600
    dpi: int = 100
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    grid: bool = True
    legend: bool = True
    band_style: Literal["line", "scatter", "cycle"] = "line"
    berry_coloring: bool = True
    component_index: int = 0
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None
    linewidth: float = 1.5
    marker_size: float = 4.0
    cmap: str = "viridis"
    berry_interpolation: bool = False
    berry_vmin: float | None = None
    berry_vmax: float | None = None
    colorbar: bool = True
    font_size: float = 10.0
    field_quantity: Literal["E2", "H2", "energy_density"] = "energy_density"


def as_dict(value: Any) -> dict[str, Any]:
    """Serialize a parameter dataclass without introducing JSON as a source of truth."""

    return asdict(value)
