import numpy as np

from legumephc import compute_berry_dipole, compute_field_observables, solve_bands, solve_berry, solve_efs
from legumephc.config import load_benchmark
from legumephc.geometry import Affine2D, Lattice2D, rotation, square_circle_spec
from legumephc.model import Model2D


def _synthetic_result(pol="te"):
    return {
        "eigenvectors": np.ones((1, 1, 2), dtype=complex),
        "frequencies": np.array([[0.2, 0.3]]),
        "gvec": np.zeros((2, 1)),
        "kpoints_cartesian": np.array([[2 * np.pi * 0.2, 0.0]]),
        "eps_inv_mat": np.ones((1, 1)),
        "polarization": pol,
    }


def test_field_observables_are_gauge_invariant_for_te_and_tm():
    spec = square_circle_spec()
    for pol in ("te", "tm"):
        result = _synthetic_result(pol)
        phases = np.exp(1j * np.array([0.31, -1.17]))
        rotated = {**result, "eigenvectors": result["eigenvectors"] * phases[None, None, :]}
        first = compute_field_observables(result, spec, lattice=Lattice2D.square(), grid_size=8)
        second = compute_field_observables(rotated, spec, lattice=Lattice2D.square(), grid_size=8)
        for key in ("E2", "H2", "energy_density", "region_energy_ratios"):
            assert np.allclose(first[key], second[key])
        assert np.allclose(first["composite_energy_density"], second["composite_energy_density"])
        assert np.allclose(first["composite_region_energy_ratios"], second["composite_region_energy_ratios"])
        assert np.allclose(np.sum(first["region_energy_ratios"], axis=-1), 1.0)


def test_phase_b_real_square_smoke_and_records(tmp_path):
    model = Model2D(square_circle_spec(), Lattice2D.square())
    qpoints = np.array([[0.2, 0.07], [-0.07, 0.2], [-0.2, -0.07], [0.07, -0.2]])
    bands = solve_bands(model, qpoints, gmax=2, numeig=4, pol="te", record_root=tmp_path)
    assert bands["path_labels"] is None
    field = compute_field_observables(bands, model.geometry, lattice=model.lattice, bands=(1, 2), grid_size=8, record_root=tmp_path)
    assert field["E2"].shape == (4, 8, 8, 2)
    efs = solve_efs(model, qpoints, gmax=2, bands=(1, 2), pol="tm", record_root=tmp_path)
    assert efs["iso_frequency_ready"] and efs["frequencies"].shape == (4, 2)
    berry = solve_berry(model, qpoints, gmax=2, bands=(1, 2), pol="te", record_root=tmp_path)
    assert berry["raw_unsymmetrized"] and berry["qualification"]["status"] == "QUALIFIED"
    assert max(edge["operator_residual"] for item in berry["covariance"] for edge in item["edges"]) < 1e-12
    multi_berry = solve_berry(model, np.stack((qpoints, qpoints + np.array([0.01, 0.01]))), gmax=2, bands=(1, 2), pol="te")
    assert multi_berry["phases"].shape == (2,) and multi_berry["areas"].shape == (2,)
    withheld = solve_berry(model, qpoints, gmax=2, bands=(1,), rank=1, pol="te", gap_floor=10.0)
    assert withheld["qualification"]["status"] == "RANK1_WITHHELD"
    assert np.isnan(withheld["qualified_curvature"]).all()
    assert len(list(tmp_path.iterdir())) == 4


def test_solve_bands_identity_path_and_frequency_record(tmp_path):
    config = load_benchmark()
    model = Model2D.from_benchmark(config, "G15")
    path = solve_bands(model, path="identity", gmax=2, numeig=2, record_root=tmp_path)
    assert path["path_labels"] == ("Gamma", "K", "M", "Gamma")
    assert path["frequencies"].shape[0] == len(path["qpoints"])
    from legumephc import frequency_at_k
    value = frequency_at_k(model, path["qpoints"][0], gmax=2, band=1, record_root=tmp_path)
    assert np.isclose(value, path["frequencies"][0, 1])
    assert len(list(tmp_path.iterdir())) == 2


def test_affine_te_tm_solver_and_mask_smoke():
    affine = Affine2D(linear=np.array([[1.0, 0.18], [0.0, 0.92]]), translation=np.array([0.07, -0.03]))
    model = Model2D(square_circle_spec(), Lattice2D.square(), affine=affine)
    assert model.point_group is None
    qpoints = np.array([[0.16, 0.09]])
    for pol in ("te", "tm"):
        solved = solve_bands(model, qpoints, gmax=2, numeig=3, pol=pol)
        observed = compute_field_observables(solved, model, bands=(1,), grid_size=8)
        assert np.isfinite(solved["frequencies"]).all()
        assert np.isfinite(observed["energy_density"]).all()


def test_berry_dipole_gradient_moment_rotation_and_uncertainty():
    qpoints = np.asarray([(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
    curvature = 2.0 * qpoints[:, 0] - 3.0 * qpoints[:, 1] + 5.0
    result = compute_berry_dipole(qpoints, curvature, uncertainty=0.1)
    assert np.allclose(result["gradient"], (2.0, -3.0))
    expected_uncertainty = np.sqrt(np.sum((qpoints / len(qpoints)) ** 2 * 0.1 ** 2, axis=0))
    assert np.allclose(result["first_moment_uncertainty"], expected_uncertainty)
    turned_q = qpoints @ rotation(90.0).T
    turned = compute_berry_dipole(turned_q, curvature)
    assert np.allclose(turned["first_moment"], result["first_moment"] @ rotation(90.0).T)
    assert result["kind"] == "geometric" and not result["physical_response"]
    response = compute_berry_dipole(qpoints, curvature, frequency_samples=np.linspace(0.0, 0.4, len(qpoints)), frequency_window=(0.1, 0.2), occupation=np.ones(len(qpoints)))
    assert response["kind"] == "physical_response" and response["physical_response"]
    assert response["selected_sample_count"] == 3
    assert np.allclose(response["first_moment"], np.mean(qpoints[2:5] * curvature[2:5, None], axis=0))
    geometric_window = compute_berry_dipole(qpoints, curvature, frequency_window=(0.1, 0.2))
    assert geometric_window["kind"] == "geometric" and not geometric_window["window_applied"]
    cartesian = compute_berry_dipole(qpoints, curvature, reciprocal_basis=np.diag([2.0, 3.0]), output_units="cartesian")
    reduced = compute_berry_dipole(qpoints, curvature)
    assert np.allclose(cartesian["first_moment"], reduced["first_moment"] @ np.diag([2.0, 3.0]).T)
    assert np.allclose(cartesian["gradient"], np.linalg.solve(np.diag([2.0, 3.0]).T, reduced["gradient"]))
    try:
        compute_berry_dipole(qpoints, curvature, output_units="cartesian")
    except ValueError as error:
        assert "reciprocal_basis" in str(error)
    else:
        raise AssertionError("Cartesian reduced-coordinate output must require a reciprocal basis")
