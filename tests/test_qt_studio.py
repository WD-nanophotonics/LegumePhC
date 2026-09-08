from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from legumephc.records import create_record
from legumephc.studio.inspection import format_significant, nearest_screen_point, record_view


def _band_record(root: Path) -> Path:
    return create_record(
        root,
        identity={"model": "Model2D", "geometry": "test", "affine": {}, "basis": {}, "solver": "test", "operation": "band_structure"},
        config={"case": {"actual_lattice_constant_m": 400e-9}, "calculation": {"band_one_based": 2}},
        summary={"operation": "band_structure", "path_labels": ["Gamma", "K", "M", "Gamma"]},
        arrays={"qpoints": np.asarray([[0.0, 0.0], [0.2, 0.1], [0.4, 0.0]]), "frequencies": np.asarray([[0.2, 0.3], [0.25, 0.35], [0.3, 0.4]])},
    )


def _berry_record(root: Path) -> Path:
    points = np.asarray([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
    return create_record(
        root,
        identity={"model": "Model2D", "geometry": "test", "affine": {}, "basis": {}, "solver": "test", "operation": "berry"},
        config={"case": {"actual_lattice_constant_m": 400e-9}, "calculation": {"berry_target_mode": "single_band", "band_one_based": 2}},
        summary={"operation": "berry", "qualification": {"per_plaquette": [{"qualified": True}] * 3 + [{"qualified": False}]}},
        arrays={"sample_centers": points, "curvature": np.asarray([-2.0, -1.0, 1.0, 2.0]), "domain_outline": np.asarray([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])},
    )


def test_significant_format_and_screen_space_snap():
    assert format_significant(0.123456789) == "0.123457"
    assert format_significant(1.2e-10) == "1.20000e-10"
    points = np.asarray([[100.0, 100.0], [300.0, 100.0]])
    assert nearest_screen_point(points, (106.0, 104.0), radius_px=10) == 0
    assert nearest_screen_point(points, (150.0, 100.0), radius_px=10) is None


def test_band_inspection_uses_raw_samples_and_current_unit(tmp_path):
    view = record_view(_band_record(tmp_path), frequency_unit="THz")
    assert view.operation == "band_structure"
    assert len(view.rows) == 6
    assert view.rows[0].values["Path"] == "Γ"
    assert np.isclose(view.rows[1].values["Frequency (THz)"], 224.8443435)
    assert view.rows[1].values["Band"] == 2


def test_berry_inspection_keeps_raw_values_and_qualification(tmp_path):
    view = record_view(_berry_record(tmp_path))
    assert [row.values["Berry curvature"] for row in view.rows] == [-2.0, -1.0, 1.0, 2.0]
    assert view.rows[-1].values["Qualified"] is False


def test_qt_window_and_interactive_canvas_offscreen(tmp_path):
    # Import Qt only after Matplotlib-backed Studio modules, matching studio.py.
    from legumephc.studio.qt_ui import QtWidgets, StudioWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = StudioWindow()
    assert window.tree.topLevelItemCount() == 3
    view = record_view(_band_record(tmp_path))
    window.result_canvas.set_record(view, {"grid": True, "legend": True, "band_line": True, "band_markers": False})
    assert len(window.result_canvas._hit_items) == 1  # hover remains available with hidden markers
    window.result_canvas.pin_row(1)
    assert window.result_canvas.pinned_rows() == [1]
    window.result_canvas.clear_pins()
    assert window.result_canvas.pinned_rows() == []
    window.dirty = False
    window.close()
    app.processEvents()


def test_qt_berry_cells_and_colorbar_offscreen(tmp_path):
    from legumephc.studio.qt_ui import QtWidgets
    from legumephc.studio.qt_viewer import ResultCanvas

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = ResultCanvas()
    canvas.set_record(record_view(_berry_record(tmp_path)), {"berry_render_mode": "sample_cells", "cmap": "RdBu_r", "colorbar": True, "grid": False})
    assert len(canvas._layer_items["Berry map"]) == 4
    assert len(canvas._layer_items["Unqualified"]) == 1
    assert len(canvas._layer_items["Colorbar"]) == 1
    canvas.close(); app.processEvents()


def test_qt_site_editor_accepts_expressions_and_adds_centered_honeycomb():
    from legumephc.motifs import triangular_motifs
    from legumephc.studio.project import new_project
    from legumephc.studio.qt_ui import QtWidgets, SitesDialog, StableComboBox

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = SitesDialog(None, new_project()["model"])
    shape = dialog.table.cellWidget(0, 1)
    assert isinstance(shape, StableComboBox)
    shape.setCurrentText("Triangle")
    dialog.table.item(0, 2).setText("root(9)/15")
    dialog.table.item(0, 3).setText("pi*0")
    dialog._add_site()
    assert dialog.table.rowCount() == 2
    expected = triangular_motifs((0.2, 0.2), (0, 0), kind="polygon", sides=3)
    for row in range(2):
        motif = dialog._row_motif(row)
        assert motif["kind"] == "polygon" and motif["sides"] == 3
        assert motif["radius"] == pytest.approx(0.2)
        assert motif["center"] == pytest.approx(expected[row]["center"])
    dialog.table.item(1, 5).setText("1/sqrt(3)")
    assert dialog._row_motif(1)["center"][1] == pytest.approx(1 / np.sqrt(3))
    dialog.close(); app.processEvents()


def test_qt_calculation_selection_is_local_and_berry_fields_are_dynamic():
    from legumephc.studio.project import add_calculation, new_project
    from legumephc.studio.qt_ui import QtCore, QtWidgets, StudioWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project = new_project()
    berry = add_calculation(project, name="Berry", operation="berry")
    project["selected_node"] = {"kind": "calculation", "id": berry["id"]}
    window = StudioWindow(project)
    window.populate = lambda: pytest.fail("calculation selection must not rebuild the full window")
    target = None
    iterator = QtWidgets.QTreeWidgetItemIterator(window.tree)
    while iterator.value():
        item = iterator.value()
        if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == ("calculation", berry["id"]):
            target = item; break
        iterator += 1
    window._tree_selected(target, None)
    window.berry_target.setCurrentIndex(window.berry_target.findData("single_band"))
    window.band.setValue(4); window._calculation_changed()
    assert window.numeig.value() == 5
    assert window.calc_form.isRowVisible(window.band)
    assert not window.calc_form.isRowVisible(window.first_band)
    window.berry_sampling.setCurrentIndex(window.berry_sampling.findData("first_bz_grid")); window._berry_controls_changed()
    assert window.calc_form.isRowVisible(window.grid_size)
    assert not window.calc_form.isRowVisible(window.qx)
    window.dirty = False; window.close(); app.processEvents()


def test_qt_geometry_expression_sync_and_stable_combobox():
    from legumephc.studio.qt_ui import QtWidgets, StableComboBox, StudioWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = StudioWindow()
    assert all(isinstance(combo, StableComboBox) for combo in window.findChildren(QtWidgets.QComboBox))
    window.actual_a.setText("root(160000)")
    window.length_unit.setCurrentText("nm")
    window.background_material.setText("sqrt(7.29)")
    window.geometry_scale.setText("1")
    window._sync_model()
    assert window.project["model"]["actual_lattice_constant_m"] == pytest.approx(400e-9)
    assert window.project["model"]["geometry"]["epsilon_background"] == pytest.approx(7.29)
    window.dirty = False; window.close(); app.processEvents()


def test_qt_real_worker_band_and_berry_smoke(tmp_path):
    from legumephc.studio.project import new_project, save_project
    from legumephc.studio.qt_ui import QtCore, QtWidgets, StudioWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project = new_project(); path = tmp_path / "smoke.legumephc-studio.json"; save_project(path, project)
    window = StudioWindow(project, path)

    def run_and_wait():
        window.run()
        assert window.process is not None
        loop = QtCore.QEventLoop(); window.process.finished.connect(loop.quit); QtCore.QTimer.singleShot(60_000, loop.quit); loop.exec()
        assert window.process is None, "worker did not finish within the smoke-test timeout"
        assert window.last_worker_error is None, window.last_worker_error

    window.operation.setCurrentIndex(window.operation.findData("band_structure")); window.samples.setValue(2); window.gmax.setValue(2); window.numeig.setValue(3); window.plot_after.setChecked(False)
    run_and_wait()
    window.operation.setCurrentIndex(window.operation.findData("berry")); window.berry_target.setCurrentIndex(window.berry_target.findData("single_band")); window.band.setValue(2); window.berry_sampling.setCurrentIndex(window.berry_sampling.findData("single_plaquette")); window.plot_after.setChecked(False)
    run_and_wait()
    assert [result["calculation_snapshot"]["operation"] for result in window.project["results"]] == ["band_structure", "berry"]
    window.dirty = False; window.close(); app.processEvents()
