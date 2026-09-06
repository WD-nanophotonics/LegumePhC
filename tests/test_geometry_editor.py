from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from legumephc.geometry import Lattice2D
from legumephc.studio.geometry_editor import GeometryExpressionError, canonical_shape, editor_value, epsilon_from_editor, model_name, safe_number, shape_choice, uniaxial_matrix
from legumephc.studio.preview import preview_geometry
from legumephc.studio.project import apply_preset, load_preset, new_preset, new_project, save_preset
from legumephc.studio.ui import _dialog_location


def test_safe_geometry_expressions_are_whitelisted():
    assert safe_number("2*pi + sqrt(9)") == pytest.approx(2 * np.pi + 3)
    for expression in ("__import__('os')", "open('x')", "foo", "sqrt()", "1/0", "nan"):
        with pytest.raises(GeometryExpressionError):
            safe_number(expression)


def test_material_representation_round_trip_and_positive_constraint():
    assert epsilon_from_editor("sqrt(9)", "n") == 9
    assert editor_value(9, "n") == 3
    with pytest.raises(GeometryExpressionError):
        epsilon_from_editor("0", "n")


def test_controlled_shape_choices_map_to_canonical_geometry():
    assert canonical_shape("Circle") == ("circle", 0)
    assert canonical_shape("Triangle") == ("polygon", 3)
    assert canonical_shape("Square") == ("polygon", 4)
    assert canonical_shape("Regular polygon", "8") == ("polygon", 8)
    assert shape_choice("polygon", 3) == "Triangle"
    assert shape_choice("polygon", 4) == "Square"
    with pytest.raises(GeometryExpressionError):
        canonical_shape("Regular polygon", "2")
    with pytest.raises(GeometryExpressionError):
        canonical_shape("cir")


def test_default_and_model_name_follow_lattice_and_motif():
    project = new_project()
    assert project["model"]["lattice"] == "triangular"
    assert project["model"]["geometry"]["center"] == [0.5, 0.0]
    assert project["model"]["geometry"]["radius"] == 0.2
    assert model_name("square", [{"kind": "polygon"}]) == "SquarePolygon"


def test_uniaxial_affine_and_reciprocal_basis():
    matrix = uniaxial_matrix(2, 30)
    assert np.linalg.det(matrix) == pytest.approx(2)
    lattice = Lattice2D.triangular()
    assert lattice.direct_basis.T @ lattice.reciprocal_basis == pytest.approx(2 * np.pi * np.eye(2))


def test_all_geometry_views_are_solver_free_and_have_equal_aspect():
    case = new_project()["model"]
    for view in ("motif", "unit_cell", "motif_array", "lattice_sites", "reciprocal_bz", "epsilon"):
        result = preview_geometry(case, view=view, size=24)
        assert result["equal_aspect"] is True
        assert result["view"] == view
    assert len(preview_geometry(case, view="motif", size=24)["motifs"]) == 1
    assert preview_geometry(case, view="reciprocal_bz", size=24)["reciprocal_identity_residual"] < 1e-12


def test_preset_round_trips_ui_state_without_results(tmp_path):
    project = new_project("preset")
    project["ui_state"]["material_representation"] = "n"
    preset = new_preset("saved")
    preset["parameters"] = {"case": project["case"], "calculation": project["calculation"], "ui_state": project["ui_state"]}
    path = save_preset(tmp_path / "saved.legumephc-preset.json", preset)
    loaded = load_preset(path)
    updated = apply_preset(new_project("target"), loaded)
    assert updated["ui_state"]["material_representation"] == "n"
    assert "results" in updated and updated["results"] == []


def test_dialog_locations_are_explicit_and_do_not_depend_on_cwd(tmp_path):
    app = SimpleNamespace(project_path=tmp_path / "project.json")
    directory, filename = _dialog_location(app, "project")
    assert Path(directory) == tmp_path and filename == "project.json"
    directory, _ = _dialog_location(SimpleNamespace(project_path=None), "preset")
    assert Path(directory).name == "presets"
    app.project = {"results": [{"id": "result-7", "record_reference": {"path": "data/.studio/records/frequency_at_k/abc"}}], "selected_result": "result-7"}
    directory, _ = _dialog_location(app, "export", record_path=app.project["selected_result"])
    assert Path(directory) == tmp_path / "data/.studio/records/frequency_at_k/abc/figures"
