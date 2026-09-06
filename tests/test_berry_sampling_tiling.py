from __future__ import annotations

import numpy as np

from legumephc.berry import first_bz_sampling
from legumephc.geometry import Lattice2D
from legumephc.studio.plotting import sample_cell_polygons


def _area(polygon: np.ndarray) -> float:
    return abs(float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))) / 2.0)


def _origin_cell(cells: list[np.ndarray], centers: np.ndarray) -> np.ndarray:
    return cells[int(np.argmin(np.linalg.norm(centers, axis=1)))]


def test_wilson_step_does_not_change_reciprocal_sample_cells():
    lattice = Lattice2D.triangular()
    fine = first_bz_sampling(lattice, grid_size=9, step=0.005)
    coarse = first_bz_sampling(lattice, grid_size=9, step=0.02)
    assert np.array_equal(fine["sample_centers"], coarse["sample_centers"])
    fine_cells = sample_cell_polygons(fine["sample_centers"], fine["domain_outline"])
    coarse_cells = sample_cell_polygons(coarse["sample_centers"], coarse["domain_outline"])
    assert all(np.allclose(left, right) for left, right in zip(fine_cells, coarse_cells))
    assert not np.allclose(fine["plaquettes"], coarse["plaquettes"])


def test_triangular_sampling_tiles_bz_with_hexagonal_interior_cells():
    sampling = first_bz_sampling(Lattice2D.triangular(), grid_size=9, step=0.005)
    cells = sample_cell_polygons(sampling["sample_centers"], sampling["domain_outline"])
    assert len(_origin_cell(cells, sampling["sample_centers"])) == 6
    assert np.isclose(sum(_area(cell) for cell in cells), _area(sampling["domain_outline"]), rtol=0, atol=1e-10)
    assert sampling["sampling_lattice"] == "triangular_reciprocal"


def test_square_sampling_tiles_bz_with_square_interior_cells():
    sampling = first_bz_sampling(Lattice2D.square(), grid_size=8, step=0.005)
    cells = sample_cell_polygons(sampling["sample_centers"], sampling["domain_outline"])
    assert len(_origin_cell(cells, sampling["sample_centers"])) == 4
    assert np.isclose(sum(_area(cell) for cell in cells), _area(sampling["domain_outline"]), rtol=0, atol=1e-10)
    assert sampling["sampling_lattice"] == "square_reciprocal"


def test_affine_sampling_uses_actual_reciprocal_basis_without_forced_hexagons():
    lattice = Lattice2D(np.asarray([[1.2, 0.35], [0.1, 0.8]]), kind="custom")
    sampling = first_bz_sampling(lattice, grid_size=8, step=0.005)
    cells = sample_cell_polygons(sampling["sample_centers"], sampling["domain_outline"])
    assert np.isclose(sum(_area(cell) for cell in cells), _area(sampling["domain_outline"]), rtol=0, atol=1e-10)
    assert sampling["sampling_lattice"] == "custom_reciprocal"
    assert not all(len(cell) == 6 for cell in cells)
