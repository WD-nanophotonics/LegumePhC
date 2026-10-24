"""Toolkit-independent Studio editing semantics (formerly tested through Tk/Qt)."""

from __future__ import annotations

import numpy as np
import pytest

from legumephc.geometry import SUBLATTICE_CENTERS
from legumephc.motifs import triangular_motifs
from legumephc.studio import forms
from legumephc.studio.preview import preview_geometry
from legumephc.studio.profile import model_from_case
from legumephc.studio.project import load_project, new_project, save_project


def test_model_form_round_trip_preserves_model_and_representation():
    project = new_project()
    form = forms.model_form(project["model"], project["ui_state"])
    assert form["background"] == "2.7"  # n view of epsilon 7.29
    applied = forms.apply_model_form(project["model"], form)
    assert applied["geometry"]["epsilon_background"] == pytest.approx(7.29)
    assert applied["affine"]["linear"] == [[1.0, 0.0], [0.0, 1.0]]
    epsilon_view = forms.model_form(applied, {"material_representation": "epsilon", "length_unit": "μm"})
    assert epsilon_view["background"] == "7.29" and epsilon_view["actual_a"] == "0.4"


def test_model_form_uniaxial_custom_and_singular_rejection():
    model = new_project()["model"]
    form = forms.model_form(model)
    uniaxial = forms.apply_model_form(model, {**form, "deformation": "uniaxial", "factor": "1.1", "angle": "30"})
    assert np.linalg.det(uniaxial["affine"]["linear"]) == pytest.approx(1.1)
    assert model_from_case(uniaxial).point_group is None  # generic affine is never assigned C3
    with pytest.raises(ValueError, match="non-singular"):
        forms.apply_model_form(model, {**form, "deformation": "custom", "affine": ["1", "2", "2", "4"]})
    with pytest.raises(ValueError):
        forms.apply_model_form(model, {**form, "geometry_scale": "-1"})


def test_second_triangular_site_becomes_honeycomb_pair():
    model = new_project()["model"]
    rows = forms.site_rows(model, "n")
    rows[0].update({"shape": "Triangle", "radius": "0.2"})
    rows = forms.add_site_row(model, rows, "n")
    assert [row["name"] for row in rows] == ["Site A", "Site B"]
    rows[1]["rotation"] = "60"
    applied = forms.apply_site_rows(model, rows, "n")
    motifs = applied["geometry"]["motifs"]
    assert len(motifs) == 2 and motifs[1]["angle_degrees"] == 60
    centers = np.asarray([motif["center"] for motif in motifs])
    expected = (SUBLATTICE_CENTERS - SUBLATTICE_CENTERS.mean(axis=0)) + np.asarray([0.5, 0.0])
    np.testing.assert_allclose(centers, expected, atol=1e-12)


def test_site_errors_name_site_and_field_and_do_not_mutate_model():
    model = new_project()["model"]
    before = forms.site_rows(model, "n")
    bad = [{**before[0], "radius": "0.3,0.2"}]
    with pytest.raises(forms.SiteError) as error:
        forms.apply_site_rows(model, bad, "n")
    assert error.value.index == 0 and error.value.field == "radius"
    assert str(error.value).startswith("Site 1 · radius")
    with pytest.raises(forms.SiteError) as error:
        forms.apply_site_rows(model, [{**before[0], "material": "0"}], "n")
    assert error.value.field == "material"
    assert forms.site_rows(model, "n") == before
    with pytest.raises(ValueError):
        forms.apply_site_rows(model, [], "n")


def test_site_expressions_and_epsilon_representation():
    model = new_project()["model"]
    row = {**forms.site_rows(model, "n")[0], "radius": "sqrt(3)/8", "material": "1.5"}
    applied = forms.apply_site_rows(model, [row], "n")
    motif = applied["geometry"]["motifs"][0]
    assert motif["radius"] == pytest.approx(np.sqrt(3) / 8)
    assert motif["epsilon"] == pytest.approx(2.25)
    assert applied["geometry"]["epsilon_inclusion"] == pytest.approx(2.25)


def test_mixed_sites_affine_and_project_roundtrip(tmp_path):
    project = new_project()
    rows = forms.site_rows({"geometry": {"motifs": triangular_motifs((0.15, 0.12))}}, "n")
    rows[1].update({"shape": "Triangle", "rotation": "60"})
    model = forms.apply_site_rows(project["model"], rows, "n")
    model = forms.apply_model_form(model, {**forms.model_form(model), "deformation": "custom", "affine": ["1.2", "0.1", "0", "0.8"], "tx": "0.1"})
    built = model_from_case(model)
    expected = np.asarray([m["center"] for m in model["geometry"]["motifs"]]) @ np.asarray([[1.2, 0.1], [0, 0.8]]).T + [0.1, 0]
    np.testing.assert_allclose(built.effective_geometry.centers, expected)
    assert built.effective_geometry.ellipse_parameters[0] is not None
    assert built.effective_geometry.transformed_vertices[1].shape == (3, 2)
    assert len(preview_geometry(model, view="motif", size=16)["motifs"]) == 2
    project["model"] = model
    loaded = load_project(save_project(tmp_path / "sites.legumephc-studio.json", project))
    assert loaded["model"]["geometry"]["motifs"] == model["geometry"]["motifs"]


def test_cell_center_per_lattice():
    model = new_project()["model"]
    assert forms.cell_center(model) == [0.5, 0.0]
    assert forms.cell_center({**model, "lattice": "square"}) == [0.5, 0.5]
    custom = {**model, "lattice": "custom", "direct_basis": [[2.0, 0.0], [0.0, 1.0]]}
    assert forms.cell_center(custom) == [1.0, 0.5]


def test_calculation_normalization_derives_bands_and_berry_numeig():
    parameters = new_project()["calculations"][0]["parameters"]
    berry = forms.normalize_calculation({**parameters, "operation": "berry", "berry_target_mode": "single_band", "band_one_based": 3})
    assert berry["composite_bands_one_based"] == [3] and berry["numeig"] == 4
    composite = forms.normalize_calculation({**parameters, "operation": "berry", "berry_target_mode": "composite_subspace", "berry_first_band": 2, "berry_last_band": 4})
    assert composite["composite_bands_one_based"] == [2, 3, 4] and composite["numeig"] == 5
    efs = forms.normalize_calculation({**parameters, "operation": "efs", "numeig": 6, "berry_first_band": 1, "berry_last_band": 2, "efs_grid_size": 7, "grid_size": 3})
    assert efs["numeig"] == 6 and efs["efs_grid_size"] == 7 and efs["grid_size"] == 3
    with pytest.raises(ValueError):
        forms.normalize_calculation({**parameters, "operation": "berry", "berry_target_mode": "composite_subspace", "berry_first_band": 4, "berry_last_band": 2})
