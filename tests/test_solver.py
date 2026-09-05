import numpy as np

from legumephc.config import load_benchmark
from legumephc.geometry import DIRECT_BASIS, geometry_spec, m7_orbit
from legumephc.solver import homogeneous_shell_frequencies, q_to_legume_k, solve_homogeneous_pwe


def test_k_conversion_is_explicit_cartesian_2pi_scaling():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(q_to_legume_k(q), 2 * np.pi * q)


def test_homogeneous_c3_shells_match():
    config = load_benchmark()
    shells = np.vstack([homogeneous_shell_frequencies(q, config.background_epsilon)[:4] for q in m7_orbit(config)])
    assert np.max(np.ptp(shells, axis=0)) < 1e-12


def test_homogeneous_te_tm_match_analytic_shells():
    config = load_benchmark()
    qpoints = m7_orbit(config)
    shells = np.vstack([homogeneous_shell_frequencies(q, config.background_epsilon)[:4] for q in qpoints])
    for pol in ("te", "tm"):
        result = solve_homogeneous_pwe(qpoints, config.background_epsilon, gmax=2, numeig=4, pol=pol)
        assert np.max(np.abs(result["frequencies"] - shells)) < 1e-12


def test_specs_need_no_mephc_runtime():
    assert geometry_spec(load_benchmark(), "G15").name == "G15"
