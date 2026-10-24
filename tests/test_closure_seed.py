"""The default point-group-closed basis must not depend on q order."""

from __future__ import annotations

import numpy as np
import pytest

from legumephc import solve_bands, solve_berry, solve_efs
from legumephc.berry import first_bz_sampling
from legumephc.diagnostics import point_group_orbit_closure
from legumephc.geometry import (
    GeometrySpec, Lattice2D, closure_center, point_group_fixed_points, point_group_operations, square_circle_spec,
)
from legumephc.model import Model2D
from legumephc.solver import solve_pwe


GMAX = 2.5


def dai_fr0() -> Model2D:
    lattice = Lattice2D.triangular(1.0)
    spec = GeometrySpec(
        name="DaiFR0", kind="polygon", radii=(0.4,), sides=(3,), angles_degrees=(0.0,), strict_c3=True,
        epsilon_background=7.0225, epsilon_inclusion=1.0, motif_kinds=("polygon",), motif_epsilons=(1.0,),
        direct_basis=lattice.direct_basis, centers=np.zeros((1, 2)),
    )
    model = Model2D(spec, lattice)
    assert model.point_group == "C3" and model.closure_qpoints is None
    return model


RECIPROCAL = 2 * np.pi * Lattice2D.triangular(1.0).reciprocal_basis_reduced


def gvec_set(gvec: np.ndarray, reciprocal: np.ndarray = RECIPROCAL) -> set[tuple[int, int]]:
    """Exact comparison: integer reciprocal-lattice indices of each G."""
    indices = np.linalg.solve(reciprocal, np.asarray(gvec, dtype=float))
    assert np.allclose(indices, np.rint(indices), atol=1e-8)
    return {tuple(int(v) for v in column) for column in np.rint(indices).astype(int).T}


def legacy_first_point_closure(model: Model2D, qpoints: np.ndarray) -> set[tuple[float, float]]:
    """The pre-fix default: close on the rotation orbit of qpoints[0]."""
    operations = point_group_operations(model.point_group)
    seed = solve_pwe(model.effective_geometry, qpoints[:1], gmax=GMAX, numeig=2, lattice=model.effective_lattice)
    orbit = np.asarray([operation @ qpoints[0] for operation in operations])
    return gvec_set(point_group_orbit_closure(seed["gvec"], orbit, list(operations), direct_basis=model.effective_lattice.direct_basis))


def test_fixed_points_and_centres():
    triangular, square = Lattice2D.triangular(1.0), Lattice2D.square(1.0)
    np.testing.assert_allclose(point_group_fixed_points(triangular, "C3"), [[0, 0], [2 / 3, 0], [4 / 3, 0]], atol=1e-12)
    np.testing.assert_allclose(point_group_fixed_points(square, "C4"), [[0, 0], [0.5, 0.5]], atol=1e-12)
    m7 = np.array([2 / 3 - 7 / 36, 0.0])
    np.testing.assert_allclose(closure_center(triangular, "C3", m7[None, :]), [2 / 3, 0], atol=1e-12)
    np.testing.assert_allclose(closure_center(triangular, "C3", np.array([[-0.62, 0.03]])), [-2 / 3, 0], atol=1e-12)
    np.testing.assert_allclose(closure_center(square, "C4", np.array([[0.47, 0.52]])), [0.5, 0.5], atol=1e-12)
    spread = first_bz_sampling(triangular, grid_size=4, step=0.02)["sample_centers"]
    np.testing.assert_array_equal(closure_center(triangular, "C3", spread), [0.0, 0.0])
    np.testing.assert_array_equal(closure_center(triangular, "C3", spread[::-1]), [0.0, 0.0])


@pytest.mark.integration
def test_old_default_depended_on_q_order_new_one_does_not():
    model = dai_fr0()
    qpoints = first_bz_sampling(model.effective_lattice, grid_size=4, step=0.02)["plaquettes"].reshape(-1, 2)
    reordered = qpoints[np.random.default_rng(7).permutation(len(qpoints))]
    assert legacy_first_point_closure(model, qpoints) != legacy_first_point_closure(model, reordered)  # the bug
    first, second = solve_bands(model, qpoints, gmax=GMAX, numeig=3), solve_bands(model, reordered, gmax=GMAX, numeig=3)
    assert gvec_set(first["gvec"]) == gvec_set(second["gvec"])
    np.testing.assert_allclose(first["closure_qpoints"], 0.0)
    order = {tuple(q): i for i, q in enumerate(np.round(qpoints, 12))}
    back = [order[tuple(q)] for q in np.round(reordered, 12)]
    np.testing.assert_allclose(second["frequencies"], np.asarray(first["frequencies"])[back], atol=1e-12)


@pytest.mark.integration
def test_default_basis_is_a_pure_rotation_closure_for_grids():
    model = dai_fr0()
    solved = solve_efs(model, gmax=GMAX, grid_size=4, bands=(0, 1), numeig=3)
    raw = solve_bands(model, solved["qpoints"], gmax=GMAX, numeig=3)
    basis = np.asarray(raw["gvec"], dtype=float)
    rotation = point_group_operations("C3")[1]
    assert gvec_set(rotation @ basis) == gvec_set(basis)


@pytest.mark.integration
def test_berry_map_is_independent_of_plaquette_order():
    model = dai_fr0()
    plaquettes = first_bz_sampling(model.effective_lattice, grid_size=3, step=0.02)["plaquettes"]
    permutation = np.random.default_rng(3).permutation(len(plaquettes))
    first = solve_berry(model, plaquettes, gmax=GMAX, bands=(1,), numeig=3)
    second = solve_berry(model, plaquettes[permutation], gmax=GMAX, bands=(1,), numeig=3)
    np.testing.assert_allclose(second["curvature"], first["curvature"][permutation], atol=1e-9)


@pytest.mark.integration
def test_orbits_near_k_keep_the_historical_basis():
    model = dai_fr0()
    seed = np.array([2 / 3 - 7 / 36, 0.0])
    orbit = np.asarray([[2 / 3, 0] + operation @ (seed - [2 / 3, 0]) for operation in point_group_operations("C3")])
    assert gvec_set(solve_bands(model, orbit, gmax=GMAX, numeig=3)["gvec"]) == legacy_first_point_closure(model, orbit)


@pytest.mark.integration
def test_c4_model_uses_m_for_local_requests_and_gamma_for_paths():
    model = Model2D(square_circle_spec(), Lattice2D.square(1.0))
    assert model.point_group == "C4"
    near_m = solve_bands(model, np.array([[0.48, 0.5], [0.5, 0.52]]), gmax=GMAX, numeig=3)
    np.testing.assert_allclose(near_m["closure_qpoints"][0], [0.5, 0.5], atol=1e-12)
    path = solve_bands(model, path="identity", gmax=GMAX, numeig=3, samples_per_segment=4)
    np.testing.assert_allclose(path["closure_qpoints"], 0.0)
