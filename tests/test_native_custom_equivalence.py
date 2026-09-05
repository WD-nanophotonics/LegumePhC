import numpy as np

from legumephc.config import load_benchmark
from legumephc.diagnostics import projector_distance
from legumephc.geometry import Lattice2D, geometry_spec
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
    result = solve_bands(model, qpoints, gmax=2, numeig=4)
    assert result["closed_gvec_count"] > result["seed_gvec_count"]
    assert np.isclose(frequency_at_k(model, qpoints[0], gmax=2, band=1), result["frequencies"][0, 1])


def test_model2d_c4_auto_uses_generic_closed_solver():
    config = load_benchmark()
    model = Model2D(geometry_spec(config, "G15"), Lattice2D.square(), point_group="C4")
    qpoints = np.array([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0], [0.0, -0.2]])
    result = solve_bands(model, qpoints, gmax=2, numeig=2)
    assert result["point_group_operations"] == 4
    assert result["closed_gvec_count"] >= result["seed_gvec_count"]
