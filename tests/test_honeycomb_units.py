import numpy as np
import pytest
from legumephc.motifs import triangular_motifs
from legumephc.geometry import SUBLATTICE_CENTERS
from legumephc.units import frequency_factor, reference_length
from legumephc.studio.project import new_project
from legumephc.studio.profile import model_from_case
from legumephc.studio.plotting import plot_record
from legumephc.records import create_record


def test_honeycomb_relative_positions_and_angles():
    motifs = triangular_motifs((.2, .18), (0, 60), kind="polygon", sides=3)
    centers = np.array([m["center"] for m in motifs])
    np.testing.assert_allclose(centers[1] - centers[0], SUBLATTICE_CENTERS[1] - SUBLATTICE_CENTERS[0])
    np.testing.assert_allclose(centers.mean(axis=0), [.5, 0])
    assert [m["angle_degrees"] for m in motifs] == [0, 60]
    assert len(triangular_motifs(.2)) == 1
    with pytest.raises(ValueError, match="Rotation"):
        triangular_motifs((.2, .18), (0,))


def test_physical_length_does_not_change_solver_model():
    case = new_project()["case"]
    before = model_from_case(case)
    case["actual_lattice_constant_m"] = 900e-9
    after = model_from_case(case)
    np.testing.assert_array_equal(before.effective_lattice.direct_basis, after.effective_lattice.direct_basis)
    assert before.geometry.radii == after.geometry.radii
    assert new_project()["ui_state"]["material_representation"] == "n"


def test_frequency_conversion_roundtrip_and_missing_length():
    length = reference_length(400, "nm")
    f = frequency_factor("THz", length)
    assert .3*f == pytest.approx(224.8443435)
    assert (.3*f)/f == pytest.approx(.3)
    assert reference_length(.4, "μm") == pytest.approx(length)
    assert frequency_factor("GHz", length) == pytest.approx(224844.3435 / .3)
    with pytest.raises(ValueError, match="not set"):
        frequency_factor("Hz", None)


def test_plot_uses_record_length(tmp_path):
    path = create_record(tmp_path, identity={"model":"Model2D", "geometry":"test", "affine":{}, "basis":{}, "solver":"test", "operation":"band_structure"}, config={"case":{"actual_lattice_constant_m":400e-9}}, summary={}, arrays={"frequencies":np.array([[.3],[.4]])})
    fig = plot_record(path, {"frequency_unit":"THz"})
    assert fig.axes[0].lines[0].get_ydata()[0] == pytest.approx(224.8443435)
    assert fig.axes[0].get_ylabel() == "Frequency (THz)"
    raw = plot_record(path)
    assert raw.axes[0].lines[0].get_ydata()[0] == pytest.approx(.3)
