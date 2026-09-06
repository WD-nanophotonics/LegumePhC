from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from legumephc.geometry import Affine2D, GeometrySpec, Lattice2D
from legumephc.model import Model2D
from legumephc.studio.preview import preview_geometry
from legumephc.studio.project import add_calculation, copy_calculation, delete_calculation, load_project, migrate_project, new_project, rename_calculation, save_project, validate_project


def test_mixed_motif_identity_and_affine_preview_preserve_order():
    case = {
        "name": "mixed",
        "lattice": "square",
        "lattice_constant": 1.0,
        "direct_basis": [[1.0, 0.0], [0.0, 1.0]],
        "geometry": {"name": "mixed", "kind": "circle", "radius": 0.12, "center": [0.25, 0.25], "epsilon_background": 7.29, "epsilon_inclusion": 1.0, "motifs": [
            {"name": "circle", "kind": "circle", "radius": 0.12, "center": [0.25, 0.25], "epsilon": 2.0},
            {"name": "polygon", "kind": "polygon", "radius": 0.16, "sides": 5, "center": [0.7, 0.7], "epsilon": 3.0},
        ]},
        "affine": {"linear": [[1.0, 0.18], [0.0, 0.92]], "translation": [0.07, -0.03]},
        "basis_policy": "native",
    }
    preview = preview_geometry(case, view="epsilon", size=24)
    assert np.isin(2.0, preview["epsilon"]) and np.isin(3.0, preview["epsilon"])
    lattice = Lattice2D.square()
    spec = GeometrySpec(name="mixed", kind="circle", radii=(.12, .16), sides=(0, 5), angles_degrees=(0., 0.), strict_c3=False, epsilon_background=7.29, epsilon_inclusion=1.0, motif_kinds=("circle", "polygon"), motif_epsilons=(2., 3.), direct_basis=lattice.direct_basis, centers=np.asarray([[.25,.25],[.7,.7]]))
    model = Model2D(spec, lattice, affine=Affine2D([[1., .18], [0., .92]], [.07, -.03]), basis_policy="native")
    assert model.effective_geometry.transformed_vertices[0] is None
    assert model.effective_geometry.transformed_vertices[1] is not None
    assert model.identity["motifs"]["epsilons"] == [2.0, 3.0]


def test_v2_roundtrip_migration_and_calculation_isolation(tmp_path):
    project = new_project("v2")
    assert "case" not in project and "calculations" in project
    add_calculation(project, name="Bands", operation="band_structure")
    copy_calculation(project, "calc-2")
    rename_calculation(project, "calc-3", "Bands copy")
    project["calculations"][1]["parameters"]["samples_per_segment"] = 8
    assert project["calculations"][0]["parameters"].get("samples_per_segment") != 8
    path = save_project(tmp_path / "project.json", project)
    raw = json.loads(path.read_text())
    assert "case" not in raw and "calculation" not in raw and "records" not in raw
    assert load_project(path)["calculations"][1]["parameters"]["samples_per_segment"] == 8

    legacy = {"schema": "legumephc-studio-project-v1", "name": "old", "case": project["model"], "calculation": project["calculations"][0]["parameters"], "plot": project["plot"], "records": [], "selected_result": None}
    migrated = migrate_project(legacy)
    assert migrated["schema"].endswith("v2") and migrated["selected_node"] == {"kind": "calculation", "id": "calc-1"}
    delete_calculation(project, "calc-3")


def test_operation_specific_grids_do_not_cross_validate():
    project = new_project("operation grids")
    efs = add_calculation(project, name="EFS", operation="efs")
    # Berry/field grid values are irrelevant to an EFS calculation.  Only the
    # dedicated EFS grid is required to be at least two.
    efs["parameters"].update({"operation": "efs", "grid_size": 1, "efs_grid_size": 6})
    validate_project(project)


def test_berry_target_mode_derives_contiguous_bands_and_eigenmode_margin():
    from legumephc.studio.worker import _selected_bands

    calculation = deepcopy(new_project()["calculations"][0]["parameters"])
    calculation["operation"] = "berry"
    calculation.update({"berry_target_mode": "single_band", "band_one_based": 2})
    assert _selected_bands(calculation) == ((1,), 3)
    calculation.update({"berry_target_mode": "composite_subspace", "berry_first_band": 2, "berry_last_band": 3})
    assert _selected_bands(calculation) == ((1, 2), 4)


def test_legacy_berry_targets_and_plot_defaults_migrate_explicitly():
    project = new_project("legacy")
    calculation = project["calculations"][0]["parameters"]
    calculation.pop("berry_target_mode")
    calculation.pop("berry_first_band")
    calculation.pop("berry_last_band")
    calculation["composite_bands_one_based"] = [2, 3]
    project["plot"].update({"berry_coloring": False})
    migrated = migrate_project(project)
    assert migrated["calculations"][0]["parameters"]["berry_target_mode"] == "composite_subspace"
    assert migrated["plot"]["berry_coloring"] is True
    assert migrated["plot"]["berry_render_mode"] == "sample_cells"


def test_epsilon_preview_retains_cartesian_equilateral_triangle():
    project = new_project("triangle")
    project["model"]["geometry"].update({
        "kind": "polygon", "radius": 0.25, "sides": 3, "center": [0.5, 0.0],
        "motifs": [{"name": "Triangle", "kind": "polygon", "radius": 0.25, "sides": 3, "center": [0.5, 0.0], "epsilon": 1.0, "angle_degrees": 0.0}],
    })
    preview = preview_geometry(project["model"], view="epsilon", size=48)
    basis = np.asarray(preview["direct_basis"])
    assert np.allclose([preview["x_grid"][0, 0], preview["y_grid"][0, 0]], [0.0, 0.0])
    assert np.allclose([preview["x_grid"][-1, -1], preview["y_grid"][-1, -1]], basis[:, 0] + basis[:, 1])
    vertices = preview["motifs"][0]["vertices"]
    lengths = np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1)
    assert np.max(lengths) - np.min(lengths) < 1e-12


def test_historical_result_snapshot_survives_calculation_operation_change():
    project = new_project("history")
    calculation = project["calculations"][0]
    band_snapshot = deepcopy(calculation["parameters"])
    band_snapshot["operation"] = "band_structure"
    project["results"].append({
        "id": "result-1",
        "calculation_id": calculation["id"],
        "record_reference": {"path": "records/band", "identity": {"operation": "band_structure"}},
        "model_snapshot": deepcopy(project["model"]),
        "calculation_snapshot": band_snapshot,
        "plot": deepcopy(project["plot"]),
    })
    calculation["operation"] = "berry"
    calculation["parameters"]["operation"] = "berry"
    validate_project(project)


def test_direct_runner_fake_operation_writes_one_record_and_default_figure(tmp_path, monkeypatch):
    from legumephc.records import create_record
    import studies.direct_runner as runner
    from studies.square.case import CASE
    from studies.square.parameters import FREQUENCY

    def fake_frequency(model, qpoint, **kwargs):
        create_record(kwargs["record_root"], identity={"model": "Model2D", "geometry": model.geometry.name, "affine": {}, "basis": {}, "solver": "fake", "operation": "frequency_at_k"}, config={}, summary={"status": "succeeded"}, arrays={"frequency": np.asarray([0.5])})
        return 0.5

    monkeypatch.setattr(runner, "frequency_at_k", fake_frequency)
    # The public API receives record_root; the fake also proves no second write
    # is introduced by plotting.
    result = runner.run_operation("square", CASE, FREQUENCY, record_root=tmp_path, show=False, testing=True)
    assert Path(result["figure_path"]).is_file()
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1


def test_direct_launchers_bootstrap_studies_without_pythonpath():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for case in ("triangular", "square", "affine"):
        launcher = root / "studies" / case / "run_frequency.py"
        result = subprocess.run([sys.executable, "-c", f"import runpy; runpy.run_path(r'{launcher}', run_name='launcher_import')"], cwd=root, env=environment, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr


def test_ui_motif_seven_column_parse_and_visibility_is_scoped():
    from legumephc.studio.ui import StudioApp, _control_text

    assert _control_text(None) == ""
    assert _control_text([0.1, 0.2]) == "0.1,0.2"

    project = new_project()
    values = {"lattice": "square", "lattice_constant": "1", "b11": "1", "b12": "0", "b21": "0", "b22": "1", "a11": "1", "a12": "0", "a21": "0", "a22": "1", "tx": "0", "ty": "0", "kind": "circle", "radius": "0.2", "sides": "8", "angle": "0"}
    app = SimpleNamespace(project=project, _vars={key: SimpleNamespace(get=lambda value=value: value) for key, value in values.items()})
    app.motif_tree = SimpleNamespace(get_children=lambda: ["motif-1"], item=lambda _item, _what: ("circle", "circle", "0.2", "[0.5, 0.5]", "2.0", "8", "0.0"))
    parsed = StudioApp._case_from_controls(app)
    assert parsed["geometry"]["motifs"][0]["epsilon"] == 2.0

    class Widget:
        def __init__(self): self.removed = False
        def grid(self): self.removed = False
        def grid_remove(self): self.removed = True
    geometry_widget, calculation_widget = Widget(), Widget()
    app._vars = {"operation": SimpleNamespace(get=lambda: "band_structure")}
    app._field_widgets = {"radius": [geometry_widget], "gmax": [calculation_widget]}
    app._calculation_field_names = {"gmax"}
    app._style_widgets = {}
    app._update_style_visibility = lambda _operation: None
    app.polarization_widget = Widget()
    app.berry_sampling_widget = Widget()
    app.berry_sampling_label = Widget()
    StudioApp._operation_changed(app)
    assert geometry_widget.removed is False and calculation_widget.removed is False
    assert app.berry_sampling_widget.removed is True and app.berry_sampling_label.removed is True


def test_calculation_grids_are_independent_from_plot_grid_boolean():
    from legumephc.studio.ui import StudioApp

    parameters = deepcopy(new_project()["calculations"][0]["parameters"])

    def controls(operation: str):
        values = {
            "operation": operation, "qx": "0.2", "qy": "0.07", "band": "2",
            "composite": "2,3", "gmax": "2", "numeig": "4", "polarization": "te",
            "field_grid": "6", "efs_grid": "7", "berry_grid": "8", "berry_step": "0.02",
            "samples": "16", "berry_sampling": "first_bz_grid", "source_result": "",
            "response_weights": "", "grid": True,
        }
        return {key: SimpleNamespace(get=lambda value=value: value) for key, value in values.items()}

    app = SimpleNamespace(_selected_calculation_entry=lambda: {"parameters": parameters})
    app._vars = controls("fields_energy")
    assert StudioApp._calculation_from_controls(app)["grid_size"] == 6
    app._vars = controls("berry")
    assert StudioApp._calculation_from_controls(app)["grid_size"] == 8
    app._vars = controls("efs")
    calculation = StudioApp._calculation_from_controls(app)
    assert calculation["efs_grid_size"] == 7
