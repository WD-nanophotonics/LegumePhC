"""K/K' half-BZ valley Chern number: domain, conventions and qualification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from legumephc.diagnostics import rank1_wilson
from legumephc.geometry import Affine2D, GeometrySpec, Lattice2D
from legumephc.model import Model2D
from legumephc.valley import _plaquettes, solve_valley_chern, valley_domain


def dai_fr0(affine: Affine2D | None = None) -> Model2D:
    lattice = Lattice2D.triangular(1.0)
    spec = GeometrySpec(
        name="DaiFR0", kind="polygon", radii=(0.4,), sides=(3,), angles_degrees=(0.0,), strict_c3=True,
        epsilon_background=7.0225, epsilon_inclusion=1.0, motif_kinds=("polygon",), motif_epsilons=(1.0,),
        direct_basis=lattice.direct_basis, centers=np.zeros((1, 2)),
    )
    return Model2D(spec, lattice, affine=affine or Affine2D())


@pytest.mark.parametrize(("delta_k", "delta_gamma", "source_count", "source_weight", "clipped_weight"), [
    (0.10, 0.10, 551, 0.4251543209876543, 0.4171355694895034),   # legacy E9F fr=0
    (0.05, 0.13, 641, 0.49459876543209874, 0.48568148019904145),  # legacy E9F fr=0.4
])
def test_domain_reproduces_legacy_e9f_quadrature(delta_k, delta_gamma, source_count, source_weight, clipped_weight):
    domain = valley_domain(Lattice2D.triangular(1.0), "K'", delta_k=delta_k, delta_gamma=delta_gamma)
    assert int(domain.retained.sum()) == source_count
    assert domain.retained.sum() / 36 ** 2 == pytest.approx(source_weight, abs=1e-15)
    assert domain.discrete_area == pytest.approx(clipped_weight, abs=1e-12)
    assert domain.continuous_area == pytest.approx(clipped_weight, abs=1e-12)
    np.testing.assert_allclose(domain.center, [-2 / 3, 0.0], atol=1e-15)


def test_untruncated_hbz_is_half_the_bz_and_valleys_are_negations():
    lattice = Lattice2D.triangular(1.0)
    k = valley_domain(lattice, "K", delta_k=0.0, delta_gamma=0.0)
    kp = valley_domain(lattice, "K'", delta_k=0.0, delta_gamma=0.0)
    assert k.continuous_area == pytest.approx(1 / math.sqrt(3), abs=1e-14)
    np.testing.assert_allclose(k.center, [2 / 3, 0.0], atol=1e-15)
    np.testing.assert_allclose(np.sort(kp.cell_centers, axis=0), np.sort(-k.cell_centers, axis=0), atol=1e-15)
    np.testing.assert_allclose(np.sort(kp.cell_weights), np.sort(k.cell_weights), atol=1e-14)


def test_plaquettes_are_counter_clockwise_and_centred():
    plaquettes = _plaquettes(np.array([[0.1, -0.2]]), 1 / 36)
    x, y = plaquettes[0, :, 0], plaquettes[0, :, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)) == pytest.approx((1 / 36) ** 2)
    np.testing.assert_allclose(plaquettes[0].mean(axis=0), [0.1, -0.2])


def test_curvature_convention_is_minus_textbook():
    """rank1_wilson flux/area equals -Omega of Xiao-Chang-Niu (A = i<u|grad u>)."""
    m, k0, h = 0.5, np.array([0.1, 0.2]), 1e-3

    def lower(k):
        values, vectors = np.linalg.eigh(np.array([[m, k[0] - 1j * k[1]], [k[0] + 1j * k[1], -m]]))
        return vectors[:, 0]

    corners = _plaquettes(k0[None, :], h)[0]
    ours = rank1_wilson([lower(c) for c in corners])["phase"] / h ** 2
    textbook = m / (2 * (k0 @ k0 + m * m) ** 1.5)
    assert ours == pytest.approx(-textbook, rel=1e-5)


def test_generic_affine_models_have_no_valleys():
    with pytest.raises(ValueError, match="verified C3"):
        solve_valley_chern(dai_fr0(Affine2D(np.array([[1.05, 0.0], [0.0, 1.0]]))), gmax=2)
    deformed = Lattice2D(np.array([[0.5, 0.5], [0.9, -0.85]]), kind="custom")
    with pytest.raises(ValueError, match="undeformed triangular"):
        valley_domain(deformed)


def test_invalid_requests_are_rejected():
    with pytest.raises(ValueError):
        valley_domain(Lattice2D.triangular(1.0), "Gamma")
    with pytest.raises(ValueError, match="whole HBZ"):
        valley_domain(Lattice2D.triangular(1.0), delta_k=1.0)
    with pytest.raises(ValueError, match="band above"):
        solve_valley_chern(dai_fr0(), gmax=2, bands=(1,), numeig=2)


@pytest.mark.integration
def test_time_reversal_and_fail_closed_reporting(tmp_path):
    model = dai_fr0()
    kp = solve_valley_chern(model, gmax=3, valley="K'", bands=(0, 1, 2), record_root=tmp_path)
    k = solve_valley_chern(model, gmax=3, valley="K", bands=(0, 1, 2))
    for left, right in zip(kp["summary"]["bands"], k["summary"]["bands"]):
        if left["valley_chern"] is not None:
            assert left["valley_chern"] == pytest.approx(-right["valley_chern"], abs=1e-6)  # TRS: C(K') = -C(K)
    band0, band1, band2 = kp["summary"]["bands"]
    assert band0["status"] == band1["status"] == "QUALIFIED_REPORTED"
    assert abs(band0["valley_chern"]) < 0.2 and 0.4 < abs(band1["valley_chern"]) < 0.8
    assert band2["status"] == "INCOMPLETE_NOT_REPORTED" and band2["valley_chern"] is None  # gap < 0.02 near M
    assert band2["SOURCE_GRID_MIDPOINT"]["failure_reasons"]["gap"] > 0
    assert kp["summary"]["cutoff"]["closed_gvec_count"] == k["summary"]["cutoff"]["closed_gvec_count"]
    assert (kp["record_path"] / "arrays.npz").is_file()


@pytest.mark.integration
def test_studio_worker_view_and_export_for_valley_study(tmp_path):
    from legumephc.studio import forms
    from legumephc.studio.document import new_workspace, write_workspace_project
    from legumephc.studio.inspection import record_view
    from legumephc.studio.plotting import plot_record
    from legumephc.studio.project import new_project
    from legumephc.studio.worker import build_worker_request, execute_request

    project = new_project("Valley")
    rows = [{"name": "Dai", "shape": "Triangle", "radius": "0.4", "rotation": "0", "center_x": "0", "center_y": "0", "material": "1", "sides": "3"}]
    project["model"] = forms.apply_site_rows(project["model"], rows, "epsilon")
    project["model"]["geometry"]["epsilon_background"] = 7.0225
    parameters = forms.normalize_calculation({**project["calculations"][0]["parameters"], "operation": "valley_chern", "berry_first_band": 1, "berry_last_band": 2, "gmax": 2.5})
    assert parameters["numeig"] == 3
    project["calculations"][0].update({"operation": "valley_chern", "parameters": parameters})
    workspace = new_workspace(tmp_path)
    result = execute_request(build_worker_request(project, write_workspace_project(workspace, project)))
    summary = result["metadata"]["summary"]
    assert summary["valley"] == "K'" and [band["band_zero_based"] for band in summary["bands"]] == [0, 1]
    view = record_view(result["record_path"], component_index=1)
    assert view.operation == "valley_chern" and len(view.rows) == 601
    assert {"Berry curvature", "Qualified", "Gap", "Source-grid node"} <= set(view.columns)
    assert sum(row.values["Source-grid node"] for row in view.rows) == 551
    figure = plot_record(result["record_path"], {"component_index": 1})
    assert "C_v" in figure.axes[0].texts[0].get_text()
