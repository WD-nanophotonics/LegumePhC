import numpy as np

from legumephc.config import load_benchmark
from legumephc.diagnostics import matrix_covariance_residual, operator_covariance_residual, projector_distance, reciprocal_c3_map
from legumephc.geometry import Lattice2D, geometry_spec, rotation, square_circle_spec
from legumephc.model import Model2D
from legumephc.solver import frequency_at_k, solve_bands, solve_pwe, solve_pwe_custom_basis


def test_native_and_custom_same_gvec_match_te_and_tm():
    config = load_benchmark()
    spec = geometry_spec(config, "G15")
    qpoints = np.array([[0.31, 0.17], [0.41, -0.08]])
    for pol in ("te", "tm"):
        native = solve_pwe(spec, qpoints, gmax=2, numeig=4, pol=pol)
        custom = solve_pwe_custom_basis(spec, qpoints, native["gvec"], numeig=4, pol=pol)
        assert np.allclose(native["eps_inv_mat"], custom["eps_inv_mat"], atol=1e-10)
        assert np.allclose(native["frequencies"], custom["frequencies"], atol=1e-10)
        for index in range(len(qpoints)):
            assert projector_distance(native["eigenvectors"][index, :, 1:3], custom["eigenvectors"][index, :, 1:3]) < 1e-8


def test_model2d_band_api_and_frequency_at_k():
    config = load_benchmark()
    model = Model2D.from_benchmark(config, "G15")
    qpoints = np.array([[0.4722222222222222, 0.0], [0.7638888888888888, -0.16839383310241256], [0.7638888888888888, 0.16839383310241256]])
    events = []
    result = solve_bands(model, qpoints, gmax=2, numeig=4, progress=events.append)
    assert result["closed_gvec_count"] > result["seed_gvec_count"]
    assert np.isclose(frequency_at_k(model, qpoints[0], gmax=2, band=1), result["frequencies"][0, 1])
    solved = [event for event in events if event["phase"] == "eigensolver" and event.get("completed")]
    assert solved[-1]["completed"] == solved[-1]["total"] == len(qpoints)


def test_model2d_c4_auto_uses_generic_closed_solver():
    model = Model2D(square_circle_spec(), Lattice2D.square())
    assert model.point_group == "C4"
    qpoints = np.array([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0], [0.0, -0.2]])
    result = solve_bands(model, qpoints, gmax=2, numeig=2)
    assert result["point_group_operations"] == 4
    assert result["closed_gvec_count"] >= result["seed_gvec_count"]


def test_square_circle_c4_basis_operator_and_projector_covariance():
    model = Model2D(square_circle_spec(), Lattice2D.square())
    rotation_90 = rotation(90.0)
    qpoints = np.array([[0.2, 0.07], [-0.07, 0.2], [-0.2, -0.07], [0.07, -0.2]])
    result = solve_bands(model, qpoints, gmax=2, numeig=4)
    basis_residuals = []
    operator_residuals = []
    projector_residuals = []
    for index in range(4):
        mapping = reciprocal_c3_map(
            result["gvec"], result["gvec"], qpoints[index], qpoints[(index + 1) % 4],
            rotation_matrix=rotation_90,
        )
        assert mapping["matched_fraction"] == 1.0
        basis_residuals.append(mapping["maximum_matching_residual"])
        operator_residuals.append(operator_covariance_residual(
            result["eps_inv_mat"], result["gvec"], qpoints[index], qpoints[(index + 1) % 4], mapping,
        ))
        assert matrix_covariance_residual(result["eps_inv_mat"], mapping) < 1e-12
        source = result["eigenvectors"][index, :, :4]
        target = result["eigenvectors"][(index + 1) % 4, :, :4]
        source_projector = source @ source.conj().T
        target_projector = target @ target.conj().T
        indices = np.asarray(mapping["indices"])
        projector_residuals.append(projector_distance(source_projector, target_projector[np.ix_(indices, indices)]))
    assert max(basis_residuals) < 1e-12
    assert max(operator_residuals) < 1e-12
    assert max(projector_residuals) < 1e-11
