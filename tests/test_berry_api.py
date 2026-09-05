import numpy as np
import pytest

from legumephc.berry import solve_berry
from legumephc.geometry import GeometrySpec, Lattice2D
from legumephc.model import Model2D


def _model():
    spec = GeometrySpec("synthetic", "circle", (0.1,), None, (0.0,), False, 2.0, 1.0, direct_basis=np.eye(2), centers=np.array([[0.5, 0.5]]))
    return Model2D(spec, Lattice2D(np.array([[1.0, 0.1], [0.0, 1.0]])))


def _fake_solver(phases=None):
    def solve(_model, qpoints, *, gmax, numeig, pol):
        qpoints = np.asarray(qpoints)
        vectors = np.zeros((len(qpoints), 3, 3), dtype=complex)
        for index in range(len(qpoints)):
            angle = 0.2 * index if phases is None else phases[index]
            vectors[index] = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]], dtype=complex)
        return {"eigenvectors": vectors, "frequencies": np.tile(np.array([[1.0, 2.0, 3.0]]), (len(qpoints), 1)), "gvec": np.zeros((2, 3)), "eps_inv_mat": np.eye(3), "polarization": pol}
    return solve


def test_multi_plaquette_area_curvature_orientation_and_gauge(monkeypatch):
    import legumephc.berry as berry
    monkeypatch.setattr(berry, "solve_bands", _fake_solver())
    first = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]])
    second = np.array([[0.5, 0.1], [0.9, 0.1], [0.9, 0.5], [0.5, 0.5]])
    result = solve_berry(_model(), np.stack((first, second)), gmax=2, bands=(0, 1), rank=2)
    assert result["phases"].shape == (2,)
    assert np.allclose(result["areas"], (0.04, 0.16))
    assert np.allclose(result["curvature"], result["phases"] / result["areas"])
    scaled = solve_berry(_model(), first[None, ...] * 2.0, gmax=2, bands=(0, 1), rank=2)
    assert np.isclose(scaled["areas"][0], 4.0 * result["areas"][0])
    assert np.isclose(scaled["phases"][0], result["phases"][0])
    assert np.isclose(scaled["curvature"][0], result["curvature"][0] / 4.0)
    with pytest.raises(ValueError, match="same orientation"):
        solve_berry(_model(), np.stack((first, second[::-1])), gmax=2, bands=(0, 1), rank=2)
    with pytest.raises(ValueError, match="nonzero"):
        solve_berry(_model(), np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]), gmax=2, bands=(0, 1), rank=2)


def test_u1_and_un_gauge_changes_do_not_change_wilson_curvature(monkeypatch):
    import legumephc.berry as berry
    base = _fake_solver()
    monkeypatch.setattr(berry, "solve_bands", base)
    plaquette = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]])
    original = solve_berry(_model(), plaquette, gmax=2, bands=(0, 1), rank=2)

    def gauged(model, qpoints, *, gmax, numeig, pol):
        output = base(model, qpoints, gmax=gmax, numeig=numeig, pol=pol)
        for index in range(len(qpoints)):
            phase = np.exp(1j * (0.4 + 0.3 * index))
            unitary = np.array([[np.cos(0.17 * index), np.sin(0.17 * index)], [-np.sin(0.17 * index), np.cos(0.17 * index)]])
            output["eigenvectors"][index, :, 0] *= phase
            output["eigenvectors"][index, :, 0:2] = output["eigenvectors"][index, :, 0:2] @ unitary
        return output
    monkeypatch.setattr(berry, "solve_bands", gauged)
    changed = solve_berry(_model(), plaquette, gmax=2, bands=(0, 1), rank=2)
    assert np.allclose(changed["phases"], original["phases"])
    assert np.allclose(changed["curvature"], original["curvature"])
    rank1 = solve_berry(_model(), plaquette, gmax=2, bands=(0,), rank=1)
    assert rank1["qualification"]["status"] in {"QUALIFIED", "RANK1_WITHHELD"}
    assert np.isfinite(rank1["curvature"]).all()
