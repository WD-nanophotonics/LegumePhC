from __future__ import annotations

import os
from pathlib import Path

import numpy as np

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
