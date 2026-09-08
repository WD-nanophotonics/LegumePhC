from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from .geometry_editor import editor_value, epsilon_from_editor, model_name
from .inspection import RecordView, record_view
from .plotting import export_figure, plot_record
from .preview import preview_geometry
from .project import (
    PRESET_SUFFIX, PROJECT_SUFFIX, add_calculation, apply_preset, copy_calculation,
    delete_calculation, load_preset, load_project, new_preset, new_project,
    record_available, record_reference, rename_calculation, save_preset, save_project,
    validate_project,
)
from .worker import EVENT_SCHEMA, build_worker_request

# Import Matplotlib-backed modules before Qt.  This avoids Qt's optional
# __feature__ import hook inspecting dateutil's legacy six importer on CPython
# 3.12 during mixed Matplotlib/Qt startup.
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .qt_viewer import ResultCanvas


ROOT = Path(__file__).resolve().parents[3]
PROJECTS_DIR = ROOT / "projects"
OPERATIONS = {
    "frequency_at_k": "Frequency at k",
    "band_structure": "Band Structure",
    "berry": "Berry Curvature",
    "efs": "EFS",
    "fields_energy": "Fields / Energy",
    "berry_curvature_dipole": "Berry Curvature Dipole",
}


def _motifs(case: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = case["geometry"]
    return deepcopy(geometry.get("motifs") or [{
        "name": geometry.get("name", "Site A"), "kind": geometry.get("kind", "circle"),
        "radius": geometry.get("radius", 0.2), "sides": geometry.get("sides", 0),
        "angle_degrees": geometry.get("angle_degrees", 0.0), "center": geometry.get("center", [0.5, 0.0]),
        "epsilon": geometry.get("epsilon_inclusion", 1.0),
    }])


class SitesDialog(QtWidgets.QDialog):
    """Structured multi-site editor; no type keywords or array syntax."""

    COLUMNS = ("Name", "Shape", "Radius r/a", "Rotation (deg)", "Center x/a", "Center y/a", "n", "Sides")

    def __init__(self, parent: QtWidgets.QWidget, case: dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("Edit lattice sites")
        self.resize(900, 360)
        self.result: list[dict[str, Any]] | None = None
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for motif in _motifs(case):
            self._add_row(motif)
        note = QtWidgets.QLabel("Geometry parameters are defined before affine transformation. Each site may use a different shape, size, rotation and material.")
        note.setWordWrap(True)
        add = QtWidgets.QPushButton("Add site")
        remove = QtWidgets.QPushButton("Remove selected")
        center = QtWidgets.QPushButton("Center selected in unit cell")
        add.clicked.connect(lambda: self._add_row({"name": f"Site {self.table.rowCount() + 1}", "kind": "circle", "radius": 0.2, "angle_degrees": 0.0, "center": [0.5, 0.0], "epsilon": 1.0, "sides": 0}))
        remove.clicked.connect(self._remove)
        center.clicked.connect(lambda: self._center(case))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        tools = QtWidgets.QHBoxLayout()
        tools.addWidget(add); tools.addWidget(remove); tools.addWidget(center); tools.addStretch()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(note); layout.addWidget(self.table); layout.addLayout(tools); layout.addWidget(buttons)

    def _add_row(self, motif: dict[str, Any]) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        shape = "Circle" if motif.get("kind", "circle") == "circle" else ({3: "Triangle", 4: "Square"}.get(int(motif.get("sides", 6)), "Regular polygon"))
        values = [motif.get("name", f"Site {row + 1}"), shape, motif.get("radius", 0.2), motif.get("angle_degrees", 0.0), *motif.get("center", [0.5, 0.0]), np.sqrt(float(motif.get("epsilon", 1.0))), motif.get("sides", 6)]
        for column, value in enumerate(values):
            if column == 1:
                combo = QtWidgets.QComboBox(); combo.addItems(["Circle", "Triangle", "Square", "Regular polygon"]); combo.setCurrentText(str(value)); self.table.setCellWidget(row, column, combo)
            else:
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(f"{float(value):.12g}" if isinstance(value, (float, np.floating)) else str(value)))

    def _remove(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _center(self, case: dict[str, Any]) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        basis = np.asarray(case.get("direct_basis"), dtype=float)
        if case.get("lattice") == "triangular":
            center = np.asarray([0.5, 0.0]) * float(case.get("lattice_constant", 1.0))
        elif case.get("lattice") == "square":
            center = np.asarray([0.5, 0.5]) * float(case.get("lattice_constant", 1.0))
        else:
            center = (basis[:, 0] + basis[:, 1]) / 2
        self.table.item(row, 4).setText(f"{center[0]:.12g}"); self.table.item(row, 5).setText(f"{center[1]:.12g}")

    def _accept(self) -> None:
        try:
            if self.table.rowCount() < 1:
                raise ValueError("add at least one site")
            motifs = []
            for row in range(self.table.rowCount()):
                shape = self.table.cellWidget(row, 1).currentText()
                kind = "circle" if shape == "Circle" else "polygon"
                sides = {"Triangle": 3, "Square": 4}.get(shape, int(float(self.table.item(row, 7).text())) if shape == "Regular polygon" else 0)
                radius = float(self.table.item(row, 2).text()); angle = float(self.table.item(row, 3).text())
                x = float(self.table.item(row, 4).text()); y = float(self.table.item(row, 5).text()); n = float(self.table.item(row, 6).text())
                if radius <= 0 or n <= 0 or (kind == "polygon" and sides < 3) or not np.isfinite([radius, angle, x, y, n]).all():
                    raise ValueError(f"row {row + 1} contains an invalid geometry value")
                motifs.append({"name": self.table.item(row, 0).text().strip() or f"Site {row + 1}", "kind": kind, "radius": radius, "sides": sides, "angle_degrees": 0.0 if kind == "circle" else angle, "center": [x, y], "epsilon": n * n})
            self.result = motifs
            self.accept()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid site", str(exc))


class InspectorTable(QtWidgets.QTableWidget):
    rowActivated = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.itemSelectionChanged.connect(self._selected)

    def set_view(self, view: RecordView | None) -> None:
        self.clear()
        if view is None:
            self.setRowCount(0); self.setColumnCount(0); return
        self.setColumnCount(len(view.columns)); self.setHorizontalHeaderLabels(view.columns); self.setRowCount(len(view.rows))
        for row_index, row in enumerate(view.rows):
            for column, name in enumerate(view.columns):
                value = row.values.get(name, "")
                text = f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value)
                self.setItem(row_index, column, QtWidgets.QTableWidgetItem(text))
        self.resizeColumnsToContents()

    def select_index(self, index: int) -> None:
        if 0 <= index < self.rowCount():
            self.blockSignals(True); self.selectRow(index); self.blockSignals(False)

    def _selected(self) -> None:
        rows = self.selectionModel().selectedRows()
        if rows:
            self.rowActivated.emit(rows[0].row())


class StudioWindow(QtWidgets.QMainWindow):
    """Qt scientific workbench for LegumePhC projects and immutable records."""

    def __init__(self, project: dict[str, Any] | None = None, project_path: str | Path | None = None):
        super().__init__()
        self.project = project or new_project()
        self.project_path = Path(project_path).resolve() if project_path else None
        self.dirty = False
        self.process: QtCore.QProcess | None = None
        self.request_path: Path | None = None
        self.active_request: dict[str, Any] | None = None
        self.stdout_buffer = ""
        self.run_started = 0.0
        self.current_view: RecordView | None = None
        self._building = False
        self._build()
        self.populate()

    def _build(self) -> None:
        self.resize(1500, 900)
        self.setWindowTitle("LegumePhC Studio")
        self._actions()
        self._tree_dock()
        self._settings_dock()
        self._bottom_dock()
        self.central_tabs = QtWidgets.QTabWidget()
        self.geometry_tabs = QtWidgets.QTabWidget()
        self.geometry_canvases: dict[str, pg.PlotWidget] = {}
        for name in ("Motif", "Unit Cell", "Motif Array", "Lattice Sites", "Reciprocal BZ", "Epsilon"):
            canvas = pg.PlotWidget(); canvas.setAspectLocked(True); canvas.showGrid(x=True, y=True, alpha=0.2)
            self.geometry_tabs.addTab(canvas, name); self.geometry_canvases[name] = canvas
        self.result_canvas = ResultCanvas()
        self.central_tabs.addTab(self.geometry_tabs, "Geometry")
        self.central_tabs.addTab(self.result_canvas, "Result")
        self.setCentralWidget(self.central_tabs)
        self.result_canvas.rowSelected.connect(self._canvas_row)
        self.result_canvas.pinsChanged.connect(self._pins_changed)
        settings = QtCore.QSettings("LegumePhC", "Studio")
        if settings.value("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        if settings.value("windowState"):
            self.restoreState(settings.value("windowState"))

    def _actions(self) -> None:
        toolbar = self.addToolBar("Main")
        for text, slot, shortcut in (("New", self.new, "Ctrl+N"), ("Open", self.open, "Ctrl+O"), ("Save", self.save, "Ctrl+S"), ("Run", self.run, "F5"), ("Cancel", self.cancel, "Esc"), ("Refresh", self.refresh, "F6"), ("Export", self.export, "Ctrl+E"), ("Copy Values", self.copy_values, "Ctrl+Shift+C"), ("Clear Pins", self.result_canvas_clear_later, "Ctrl+Shift+X")):
            action = QtGui.QAction(text, self); action.triggered.connect(slot); action.setShortcut(shortcut); toolbar.addAction(action)
        file_menu = self.menuBar().addMenu("Project")
        for action in toolbar.actions()[:3]: file_menu.addAction(action)
        file_menu.addSeparator(); file_menu.addAction("Save As…", self.save_as)
        preset_menu = file_menu.addMenu("Parameter Preset")
        preset_menu.addAction("Apply…", self.apply_preset_file); preset_menu.addAction("Save…", self.save_preset_file)
        file_menu.addSeparator(); file_menu.addAction("Exit", self.close)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Reset Plot View", lambda: self.result_canvas.plot.autoRange())

    def result_canvas_clear_later(self) -> None:
        if hasattr(self, "result_canvas"): self.result_canvas.clear_pins()

    def _tree_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Project", self); dock.setObjectName("ProjectDock")
        self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderHidden(True); self.tree.currentItemChanged.connect(self._tree_selected)
        dock.setWidget(self.tree); self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _settings_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Settings", self); dock.setObjectName("SettingsDock")
        self.settings_tabs = QtWidgets.QTabWidget(); dock.setWidget(self.settings_tabs)
        self.model_page = QtWidgets.QWidget(); self.calc_page = QtWidgets.QWidget(); self.plot_page = QtWidgets.QWidget()
        self.settings_tabs.addTab(self.model_page, "Geometry"); self.settings_tabs.addTab(self.calc_page, "Calculation"); self.settings_tabs.addTab(self.plot_page, "Plot Style")
        self._build_model_form(); self._build_calc_form(); self._build_plot_form()
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _bottom_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Data / Progress / Log", self); dock.setObjectName("BottomDock")
        tabs = QtWidgets.QTabWidget()
        self.inspector = InspectorTable(); self.inspector.rowActivated.connect(self.result_canvas_select_later)
        progress_page = QtWidgets.QWidget(); progress_layout = QtWidgets.QVBoxLayout(progress_page)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100)
        self.progress_label = QtWidgets.QLabel("Idle")
        progress_layout.addWidget(self.progress_label); progress_layout.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.result_details = QtWidgets.QPlainTextEdit(); self.result_details.setReadOnly(True)
        self.layers = QtWidgets.QWidget(); self.layers_layout = QtWidgets.QVBoxLayout(self.layers); self.layers_layout.addStretch()
        tabs.addTab(self.inspector, "Data Inspector"); tabs.addTab(self.layers, "Layers"); tabs.addTab(self.result_details, "Result Details"); tabs.addTab(progress_page, "Progress"); tabs.addTab(self.log, "Log")
        dock.setWidget(tabs); self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def result_canvas_select_later(self, index: int) -> None:
        if hasattr(self, "result_canvas"): self.result_canvas.select_row(index)

    def _build_model_form(self) -> None:
        form = QtWidgets.QFormLayout(self.model_page)
        self.lattice = QtWidgets.QComboBox(); self.lattice.addItems(["triangular", "square", "custom"])
        self.actual_a = QtWidgets.QDoubleSpinBox(); self.actual_a.setRange(1e-6, 1e9); self.actual_a.setDecimals(9); self.actual_a.setSuffix(" nm")
        self.background_n = QtWidgets.QDoubleSpinBox(); self.background_n.setRange(0.001, 100); self.background_n.setDecimals(9)
        self.basis_policy = QtWidgets.QComboBox(); self.basis_policy.addItems(["auto", "native"])
        self.deformation = QtWidgets.QComboBox(); self.deformation.addItems(["none", "uniaxial", "custom"])
        self.factor = QtWidgets.QDoubleSpinBox(); self.factor.setRange(0.001, 100); self.factor.setDecimals(9)
        self.angle = QtWidgets.QDoubleSpinBox(); self.angle.setRange(-360, 360); self.angle.setDecimals(9); self.angle.setSuffix("°")
        self.sites_button = QtWidgets.QPushButton("Edit sites…"); self.sites_button.clicked.connect(self.edit_sites)
        form.addRow("Lattice", self.lattice); form.addRow("Actual lattice constant", self.actual_a); form.addRow("Background n", self.background_n); form.addRow("Basis policy", self.basis_policy); form.addRow("Sites", self.sites_button); form.addRow("Deformation", self.deformation); form.addRow("Stretch factor", self.factor); form.addRow("Stretch angle", self.angle)
        for widget in (self.lattice, self.actual_a, self.background_n, self.basis_policy, self.deformation, self.factor, self.angle):
            signal = widget.currentTextChanged if isinstance(widget, QtWidgets.QComboBox) else widget.valueChanged
            signal.connect(self._model_changed)

    def _build_calc_form(self) -> None:
        form = QtWidgets.QFormLayout(self.calc_page)
        self.operation = QtWidgets.QComboBox(); [self.operation.addItem(label, key) for key, label in OPERATIONS.items()]
        self.qx = QtWidgets.QDoubleSpinBox(); self.qy = QtWidgets.QDoubleSpinBox()
        for widget in (self.qx, self.qy): widget.setRange(-100, 100); widget.setDecimals(9)
        self.band = QtWidgets.QSpinBox(); self.band.setRange(1, 100)
        self.first_band = QtWidgets.QSpinBox(); self.first_band.setRange(1, 100)
        self.last_band = QtWidgets.QSpinBox(); self.last_band.setRange(1, 100)
        self.gmax = QtWidgets.QDoubleSpinBox(); self.gmax.setRange(0.01, 100); self.gmax.setDecimals(6)
        self.numeig = QtWidgets.QSpinBox(); self.numeig.setRange(1, 200)
        self.pol = QtWidgets.QComboBox(); self.pol.addItems(["te", "tm"])
        self.samples = QtWidgets.QSpinBox(); self.samples.setRange(2, 10000)
        self.grid_size = QtWidgets.QSpinBox(); self.grid_size.setRange(2, 1000)
        self.efs_grid = QtWidgets.QSpinBox(); self.efs_grid.setRange(2, 1000)
        self.berry_step = QtWidgets.QDoubleSpinBox(); self.berry_step.setRange(1e-8, 1); self.berry_step.setDecimals(9)
        for label, widget in (("Calculation", self.operation), ("qₓ", self.qx), ("qᵧ", self.qy), ("Target band", self.band), ("Composite first band", self.first_band), ("Composite last band", self.last_band), ("gmax", self.gmax), ("Eigenvalues", self.numeig), ("Polarization", self.pol), ("Samples per segment", self.samples), ("Berry grid size", self.grid_size), ("EFS grid size", self.efs_grid), ("Berry step", self.berry_step)):
            form.addRow(label, widget)
        calculation_tools = QtWidgets.QHBoxLayout()
        for text, slot in (("Add", self.add_calculation), ("Copy", self.copy_calculation), ("Rename", self.rename_calculation), ("Delete", self.delete_calculation)):
            button = QtWidgets.QPushButton(text); button.clicked.connect(slot); calculation_tools.addWidget(button)
        form.addRow(calculation_tools)
        self.operation.currentIndexChanged.connect(self._calculation_changed)
        for widget in (self.qx, self.qy, self.band, self.first_band, self.last_band, self.gmax, self.numeig, self.pol, self.samples, self.grid_size, self.efs_grid, self.berry_step):
            signal = widget.currentTextChanged if isinstance(widget, QtWidgets.QComboBox) else widget.valueChanged
            signal.connect(self._calculation_changed)

    def _build_plot_form(self) -> None:
        form = QtWidgets.QFormLayout(self.plot_page)
        self.unit = QtWidgets.QComboBox(); self.unit.addItems(["Normalized", "GHz", "THz"])
        self.band_lines = QtWidgets.QCheckBox(); self.band_markers = QtWidgets.QCheckBox(); self.grid = QtWidgets.QCheckBox(); self.legend = QtWidgets.QCheckBox(); self.colorbar = QtWidgets.QCheckBox(); self.sample_centers = QtWidgets.QCheckBox()
        self.render_mode = QtWidgets.QComboBox(); self.render_mode.addItem("Sample-cell tiling", "sample_cells"); self.render_mode.addItem("Linear interpolation", "linear_interpolation")
        self.cmap = QtWidgets.QComboBox(); self.cmap.setEditable(True); self.cmap.addItems(["RdBu_r", "viridis", "plasma", "magma", "coolwarm"])
        self.width = QtWidgets.QSpinBox(); self.width.setRange(100, 10000); self.height = QtWidgets.QSpinBox(); self.height.setRange(100, 10000); self.dpi = QtWidgets.QSpinBox(); self.dpi.setRange(30, 1200)
        for label, widget in (("Frequency unit", self.unit), ("Band lines", self.band_lines), ("Band markers", self.band_markers), ("Grid", self.grid), ("Legend", self.legend), ("Berry render", self.render_mode), ("Colormap", self.cmap), ("Colorbar", self.colorbar), ("Sample centers", self.sample_centers), ("Export width", self.width), ("Export height", self.height), ("DPI", self.dpi)):
            form.addRow(label, widget)
        apply = QtWidgets.QPushButton("Apply to Current Plot"); apply.clicked.connect(self.plot_selected); form.addRow(apply)
        for widget in (self.unit, self.band_lines, self.band_markers, self.grid, self.legend, self.render_mode, self.cmap, self.colorbar, self.sample_centers, self.width, self.height, self.dpi):
            signal = widget.currentTextChanged if isinstance(widget, QtWidgets.QComboBox) else (widget.toggled if isinstance(widget, QtWidgets.QCheckBox) else widget.valueChanged)
            signal.connect(self._plot_changed)

    def populate(self) -> None:
        self._building = True
        model = self.project["model"]; geometry = model["geometry"]; deformation = model.get("deformation", {})
        self.lattice.setCurrentText(model.get("lattice", "triangular")); self.actual_a.setValue(float(model.get("actual_lattice_constant_m") or 400e-9) * 1e9); self.background_n.setValue(np.sqrt(float(geometry.get("epsilon_background", 7.29)))); self.basis_policy.setCurrentText(model.get("basis_policy", "auto")); self.deformation.setCurrentText(deformation.get("kind", "none")); self.factor.setValue(float(deformation.get("factor", 1))); self.angle.setValue(float(deformation.get("angle_degrees", 0)))
        calculation = self._selected_calculation()["parameters"]
        index = self.operation.findData(calculation.get("operation")); self.operation.setCurrentIndex(max(0, index)); self.qx.setValue(float(calculation.get("qpoint", [0, 0])[0])); self.qy.setValue(float(calculation.get("qpoint", [0, 0])[1])); self.band.setValue(int(calculation.get("band_one_based", 2))); self.first_band.setValue(int(calculation.get("berry_first_band", 2))); self.last_band.setValue(int(calculation.get("berry_last_band", 3))); self.gmax.setValue(float(calculation.get("gmax", 2))); self.numeig.setValue(int(calculation.get("numeig", 3))); self.pol.setCurrentText(calculation.get("polarization", "te")); self.samples.setValue(int(calculation.get("samples_per_segment", 16))); self.grid_size.setValue(int(calculation.get("grid_size", 8))); self.efs_grid.setValue(int(calculation.get("efs_grid_size", 5))); self.berry_step.setValue(float(calculation.get("berry_step", .02)))
        style = self.project["plot"]; self.unit.setCurrentText(style.get("frequency_unit", "Normalized")); self.band_lines.setChecked(bool(style.get("band_line", True))); self.band_markers.setChecked(bool(style.get("band_markers", False))); self.grid.setChecked(bool(style.get("grid", True))); self.legend.setChecked(bool(style.get("legend", True))); self.colorbar.setChecked(bool(style.get("colorbar", True))); self.sample_centers.setChecked(bool(style.get("show_sample_centers", False))); self.render_mode.setCurrentIndex(max(0, self.render_mode.findData(style.get("berry_render_mode", "sample_cells")))); self.cmap.setCurrentText(style.get("cmap", "RdBu_r")); self.width.setValue(int(style.get("width_px", 900))); self.height.setValue(int(style.get("height_px", 600))); self.dpi.setValue(int(style.get("dpi", 100)))
        self._building = False
        self._populate_tree(); self.refresh_geometry(); self._update_title()

    def _selected_calculation(self) -> dict[str, Any]:
        selected = self.project.get("selected_node", {}).get("id")
        return next((item for item in self.project["calculations"] if item["id"] == selected), self.project["calculations"][0])

    def _populate_tree(self) -> None:
        self.tree.clear(); model = QtWidgets.QTreeWidgetItem(["Model / Geometry"]); model.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("geometry", "geometry")); self.tree.addTopLevelItem(model)
        calculations = QtWidgets.QTreeWidgetItem(["Calculations"]); self.tree.addTopLevelItem(calculations)
        for entry in self.project["calculations"]:
            item = QtWidgets.QTreeWidgetItem([entry.get("name", OPERATIONS.get(entry["operation"], entry["operation"]))]); item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("calculation", entry["id"])); calculations.addChild(item)
        results = QtWidgets.QTreeWidgetItem(["Results"]); self.tree.addTopLevelItem(results)
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for result in reversed(self.project.get("results", [])):
            operation = result.get("calculation_snapshot", {}).get("operation", "result")
            group = groups.get(operation)
            if group is None:
                group = QtWidgets.QTreeWidgetItem([OPERATIONS.get(operation, operation)]); results.addChild(group); groups[operation] = group
            path = result.get("record_reference", {}).get("path", "unavailable")
            item = QtWidgets.QTreeWidgetItem([Path(path).name]); item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("result", result["id"])); group.addChild(item)
        self.tree.expandAll()

    def _tree_selected(self, current, _previous) -> None:
        if current is None or current.data(0, QtCore.Qt.ItemDataRole.UserRole) is None: return
        kind, identity = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self.project["selected_node"] = {"kind": kind, "id": identity}
        if kind == "geometry": self.settings_tabs.setCurrentWidget(self.model_page); self.central_tabs.setCurrentWidget(self.geometry_tabs)
        elif kind == "calculation": self.settings_tabs.setCurrentWidget(self.calc_page); self.populate()
        else:
            result = next(item for item in self.project["results"] if item["id"] == identity); self.project["selected_result"] = result["record_reference"]["path"]; self.plot_selected(); self.central_tabs.setCurrentWidget(self.result_canvas)

    def _sync_model(self) -> None:
        model = self.project["model"]; model["lattice"] = self.lattice.currentText(); model["actual_lattice_constant_m"] = self.actual_a.value() * 1e-9; model["geometry"]["epsilon_background"] = self.background_n.value() ** 2; model["basis_policy"] = self.basis_policy.currentText(); model.setdefault("deformation", {}).update({"kind": self.deformation.currentText(), "factor": self.factor.value(), "angle_degrees": self.angle.value()}); model["name"] = model_name(model["lattice"], _motifs(model)); model["geometry"]["name"] = model["name"]

    def _sync_calculation(self) -> None:
        calc = self._selected_calculation(); value = calc["parameters"]; operation = self.operation.currentData(); value.update({"operation": operation, "qpoint": [self.qx.value(), self.qy.value()], "band_one_based": self.band.value(), "berry_first_band": self.first_band.value(), "berry_last_band": self.last_band.value(), "gmax": self.gmax.value(), "numeig": self.numeig.value(), "polarization": self.pol.currentText(), "samples_per_segment": self.samples.value(), "grid_size": self.grid_size.value(), "efs_grid_size": self.efs_grid.value(), "berry_step": self.berry_step.value()}); value["composite_bands_one_based"] = list(range(self.first_band.value(), self.last_band.value() + 1)); value["berry_target_mode"] = "single_band" if self.first_band.value() == self.last_band.value() == self.band.value() else "composite_subspace"; calc["operation"] = operation

    def _sync_plot(self) -> None:
        self.project["plot"].update({"frequency_unit": self.unit.currentText(), "band_line": self.band_lines.isChecked(), "band_markers": self.band_markers.isChecked(), "grid": self.grid.isChecked(), "legend": self.legend.isChecked(), "berry_render_mode": self.render_mode.currentData(), "cmap": self.cmap.currentText(), "colorbar": self.colorbar.isChecked(), "show_sample_centers": self.sample_centers.isChecked(), "width_px": self.width.value(), "height_px": self.height.value(), "dpi": self.dpi.value()})

    def _model_changed(self, *_args) -> None:
        if self._building: return
        self._sync_model(); self.dirty = True; self.statusBar().showMessage("Geometry changed — pending Refresh/Run")

    def _calculation_changed(self, *_args) -> None:
        if self._building: return
        self._sync_calculation(); self.dirty = True; self.statusBar().showMessage("Calculation changed")

    def _plot_changed(self, *_args) -> None:
        if self._building: return
        self._sync_plot(); self.dirty = True; self.statusBar().showMessage("Pending plot update")

    def edit_sites(self) -> None:
        self._sync_model(); dialog = SitesDialog(self, self.project["model"])
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.result is not None:
            geometry = self.project["model"]["geometry"]; geometry["motifs"] = dialog.result; geometry["kind"] = dialog.result[0]["kind"]; geometry["radius"] = dialog.result[0]["radius"]; geometry["sides"] = dialog.result[0]["sides"]; geometry["angle_degrees"] = dialog.result[0]["angle_degrees"]; geometry["center"] = dialog.result[0]["center"]; geometry["epsilon_inclusion"] = dialog.result[0]["epsilon"]; self.dirty = True; self.refresh_geometry()

    def refresh_geometry(self) -> None:
        self._sync_model()
        mapping = {"Motif": "motif", "Unit Cell": "unit_cell", "Motif Array": "motif_array", "Lattice Sites": "lattice_sites", "Reciprocal BZ": "reciprocal_bz", "Epsilon": "epsilon"}
        try:
            for name, kind in mapping.items(): self._render_preview(self.geometry_canvases[name], preview_geometry(self.project["model"], view=kind, size=96))
            self.statusBar().showMessage("Geometry preview current")
        except Exception as exc: self.statusBar().showMessage(f"Preview failed: {exc}")

    @staticmethod
    def _render_preview(canvas: pg.PlotWidget, data: dict[str, Any]) -> None:
        canvas.clear(); canvas.setAspectLocked(True)
        if data.get("epsilon") is not None:
            x, y, epsilon = np.asarray(data["x_grid"]), np.asarray(data["y_grid"]), np.asarray(data["epsilon"])
            mesh = pg.PColorMeshItem(x, y, epsilon[:-1, :-1], colorMap=pg.colormap.getFromMatplotlib("viridis")); canvas.addItem(mesh)
        elif data.get("points") is not None:
            points = np.asarray(data["points"])
            if points.ndim == 2: canvas.addItem(pg.ScatterPlotItem(pos=points, size=7, brush=pg.mkBrush("#8ecae6"), pen=pg.mkPen("#219ebc")))
        if data.get("bz_vertices") is not None:
            vertices = np.asarray(data["bz_vertices"]); closed = np.vstack([vertices, vertices[0]]); canvas.plot(closed[:, 0], closed[:, 1], pen=pg.mkPen("k", width=2))
        if data.get("direct_basis") is not None and data.get("view") in {"lattice_sites", "unit_cell"}:
            basis = np.asarray(data["direct_basis"])
            for column, color in ((0, "r"), (1, "b")): canvas.plot([0, basis[0, column]], [0, basis[1, column]], pen=pg.mkPen(color, width=2))
        canvas.autoRange()

    def refresh(self) -> None:
        self.refresh_geometry()
        if self.project.get("selected_result"): self.plot_selected()

    def _selected_result_entry(self) -> dict[str, Any] | None:
        selected = self.project.get("selected_result")
        return next((item for item in self.project.get("results", []) if item.get("record_reference", {}).get("path") == selected or item.get("id") == selected), None)

    def plot_selected(self) -> None:
        self._sync_plot(); result = self._selected_result_entry()
        if result is None or self.project_path is None: return
        available, reason = record_available(self.project_path.parent, result["record_reference"])
        if not available: self.statusBar().showMessage(f"Result unavailable: {reason}"); return
        path = self.project_path.parent / result["record_reference"]["path"]
        try:
            self.current_view = record_view(path, frequency_unit=self.project["plot"].get("frequency_unit", "Normalized"), component_index=int(self.project["plot"].get("component_index", 0)), field_quantity=self.project["plot"].get("field_quantity", "energy_density"))
            pins = result.get("display_state", {}).get("pins", [])
            self.result_canvas.set_record(self.current_view, self.project["plot"], pins=pins); self.inspector.set_view(self.current_view); self._populate_layers(); self.result_details.setPlainText(json.dumps({"record": str(path), "operation": self.current_view.operation, "summary": self.current_view.summary, "calculation": result.get("calculation_snapshot"), "model": result.get("model_snapshot")}, indent=2, ensure_ascii=False, default=str)); self.central_tabs.setCurrentWidget(self.result_canvas); self.statusBar().showMessage(f"{self.current_view.operation} · {path}")
        except Exception as exc: QtWidgets.QMessageBox.warning(self, "Plot failed", str(exc))

    def _canvas_row(self, index: int) -> None:
        self.inspector.select_index(index); self.result_canvas.select_row(index)
        if self.current_view and 0 <= index < len(self.current_view.rows): self.statusBar().showMessage(self.current_view.rows[index].tooltip().replace("\n", " · "))

    def _pins_changed(self, pins: list[int]) -> None:
        result = self._selected_result_entry()
        if result is not None: result.setdefault("display_state", {})["pins"] = list(pins); self.dirty = True

    def copy_values(self) -> None:
        rows = self.inspector.selectionModel().selectedRows()
        if rows and self.current_view: QtWidgets.QApplication.clipboard().setText(self.current_view.rows[rows[0].row()].tooltip())

    def _populate_layers(self) -> None:
        while self.layers_layout.count() > 1:
            item = self.layers_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for name in self.result_canvas.layer_names():
            check = QtWidgets.QCheckBox(name); check.setChecked(True); check.toggled.connect(lambda visible, layer=name: self.result_canvas.set_layer_visible(layer, visible)); self.layers_layout.insertWidget(self.layers_layout.count() - 1, check)

    def add_calculation(self) -> None:
        entry = add_calculation(self.project, name="New calculation", operation="frequency_at_k")
        self.project["selected_node"] = {"kind": "calculation", "id": entry["id"]}; self.dirty = True; self.populate()

    def copy_calculation(self) -> None:
        source = self._selected_calculation(); entry = copy_calculation(self.project, source["id"])
        self.project["selected_node"] = {"kind": "calculation", "id": entry["id"]}; self.dirty = True; self.populate()

    def rename_calculation(self) -> None:
        entry = self._selected_calculation(); name, ok = QtWidgets.QInputDialog.getText(self, "Rename calculation", "Name", text=entry.get("name", "Calculation"))
        if ok and name.strip(): rename_calculation(self.project, entry["id"], name.strip()); self.dirty = True; self._populate_tree()

    def delete_calculation(self) -> None:
        try: delete_calculation(self.project, self._selected_calculation()["id"]); self.dirty = True; self.populate()
        except ValueError as exc: QtWidgets.QMessageBox.warning(self, "Cannot delete calculation", str(exc))

    def apply_preset_file(self) -> None:
        directory = self.project_path.parent / "presets" if self.project_path else ROOT / "presets"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Apply parameter preset", str(directory), f"LegumePhC preset (*{PRESET_SUFFIX});;JSON (*.json)")
        if path: self.project = apply_preset(self.project, load_preset(path)); self.dirty = True; self.populate()

    def save_preset_file(self) -> None:
        directory = self.project_path.parent / "presets" if self.project_path else ROOT / "presets"; directory.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save parameter preset", str(directory / f"Untitled{PRESET_SUFFIX}"), f"LegumePhC preset (*{PRESET_SUFFIX})")
        if path:
            self._sync_model(); self._sync_calculation(); preset = new_preset(Path(path).stem); preset["parameters"] = {"case": deepcopy(self.project["model"]), "calculation": deepcopy(self._selected_calculation()["parameters"]), "ui_state": deepcopy(self.project.get("ui_state", {}))}; save_preset(path, preset)

    def new(self) -> None:
        if not self._confirm_discard(): return
        self.project = new_project(); self.project_path = None; self.dirty = True; self.current_view = None; self.inspector.set_view(None); self.populate()

    def open(self) -> None:
        if not self._confirm_discard(): return
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", str(self.project_path.parent if self.project_path else PROJECTS_DIR), f"LegumePhC project (*{PROJECT_SUFFIX});;JSON (*.json)")
        if path: self.project = load_project(path); self.project_path = Path(path).resolve(); self.dirty = False; self.populate(); self.plot_selected()

    def save(self) -> bool:
        if self.project_path is None: return self.save_as()
        try:
            self._sync_model(); self._sync_calculation(); self._sync_plot(); save_project(self.project_path, self.project); self.dirty = False; self._update_title(); return True
        except Exception as exc: QtWidgets.QMessageBox.warning(self, "Save failed", str(exc)); return False

    def save_as(self) -> bool:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", str((self.project_path or PROJECTS_DIR / f"Untitled{PROJECT_SUFFIX}")), f"LegumePhC project (*{PROJECT_SUFFIX})")
        if not path: return False
        self.project_path = Path(path if path.endswith(PROJECT_SUFFIX) else path + PROJECT_SUFFIX).resolve(); return self.save()

    def export(self) -> None:
        result = self._selected_result_entry()
        if result is None or self.project_path is None: return
        path = self.project_path.parent / result["record_reference"]["path"]
        figures = path / "figures"; figures.mkdir(parents=True, exist_ok=True)
        target, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export current view", str(figures / "result.png"), "Images (*.png *.pdf *.svg)")
        if not target: return
        style = dict(self.project["plot"]); style["x_limits"], style["y_limits"] = self.result_canvas.view_limits()
        figure = plot_record(path, style); export_figure(figure, target, width_px=style["width_px"], height_px=style["height_px"], dpi=style["dpi"]); self.statusBar().showMessage(f"Exported {target}")

    def run(self) -> None:
        if self.process is not None: return
        try:
            if self.project_path is None and not self.save_as(): return
            self._sync_model(); self._sync_calculation(); self._sync_plot(); validate_project(self.project); self.save()
            request = build_worker_request(self.project, self.project_path); handle, name = tempfile.mkstemp(prefix="legumephc-qt-", suffix=".json"); import os; os.close(handle); self.request_path = Path(name); self.request_path.write_text(json.dumps(request), encoding="utf-8"); self.active_request = deepcopy(request)
            self.process = QtCore.QProcess(self); self.process.setWorkingDirectory(str(ROOT)); self.process.setProgram(sys.executable); self.process.setArguments(["-m", "legumephc.studio.worker", "--request", str(self.request_path)]); self.process.readyReadStandardOutput.connect(self._read_worker); self.process.readyReadStandardError.connect(self._read_worker_error); self.process.finished.connect(self._worker_finished); self.stdout_buffer = ""; self.run_started = time.monotonic(); self.progress.setRange(0, 0); self.progress_label.setText(f"Starting {request['calculation']['operation']}"); self.process.start()
        except Exception as exc: QtWidgets.QMessageBox.warning(self, "Run failed", str(exc)); self._cleanup_worker()

    def _read_worker(self) -> None:
        if self.process is None: return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            try: event = json.loads(line)
            except json.JSONDecodeError: self.log.appendPlainText(line); continue
            if event.get("schema") != EVENT_SCHEMA: continue
            if event.get("completed") is not None and event.get("total"):
                total = int(event["total"]); completed = int(event["completed"]); self.progress.setRange(0, total); self.progress.setValue(completed); self.progress_label.setText(f"{event.get('phase', event['event'])}: {completed}/{total} · {time.monotonic() - self.run_started:.1f} s")
            else: self.progress.setRange(0, 0); self.progress_label.setText(str(event.get("message", event["event"])))
            if event.get("event") in {"phase", "record_written", "failed"}: self.log.appendPlainText(str(event.get("message", event.get("error", event["event"]))))
            if event.get("event") == "completed": self._accept_worker_result(event)

    def _read_worker_error(self) -> None:
        if self.process is not None: self.log.appendPlainText(bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip())

    def _accept_worker_result(self, event: dict[str, Any]) -> None:
        assert self.project_path is not None and self.active_request is not None
        reference = record_reference(Path(event["record_path"]), self.project_path.parent); calculation_id = self.active_request.get("calculation_id", self._selected_calculation()["id"]); result_id = f"result-{len(self.project.get('results', [])) + 1}"
        self.project["results"].append({"id": result_id, "calculation_id": calculation_id, "record_reference": reference, "model_snapshot": self.active_request["model_snapshot"], "calculation_snapshot": self.active_request["calculation_snapshot"], "plot": deepcopy(self.project["plot"]), "display_state": {"pins": []}}); self.project["selected_result"] = reference["path"]; self.dirty = True; self._populate_tree(); self.plot_selected()

    def _worker_finished(self, exit_code: int, _status) -> None:
        if exit_code != 0: self.progress_label.setText(f"Worker failed with exit code {exit_code}")
        else: self.progress.setRange(0, 100); self.progress.setValue(100); self.progress_label.setText(f"Completed in {time.monotonic() - self.run_started:.1f} s")
        self._cleanup_worker()

    def cancel(self) -> None:
        if self.process is None: return
        self.process.terminate()
        if not self.process.waitForFinished(5000): self.process.kill(); self.process.waitForFinished(5000)
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress_label.setText("Cancelled; exact worker terminated"); self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self.request_path is not None: self.request_path.unlink(missing_ok=True)
        self.request_path = None; self.process = None; self.active_request = None

    def _confirm_discard(self) -> bool:
        if not self.dirty: return True
        answer = QtWidgets.QMessageBox.question(self, "Unsaved project", "Save changes before continuing?", QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard | QtWidgets.QMessageBox.StandardButton.Cancel)
        return self.save() if answer == QtWidgets.QMessageBox.StandardButton.Save else answer == QtWidgets.QMessageBox.StandardButton.Discard

    def _update_title(self) -> None:
        self.setWindowTitle(f"{'*' if self.dirty else ''}{self.project.get('name', 'Untitled')} — LegumePhC Studio")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._confirm_discard(): event.ignore(); return
        if self.process is not None: self.cancel()
        settings = QtCore.QSettings("LegumePhC", "Studio"); settings.setValue("geometry", self.saveGeometry()); settings.setValue("windowState", self.saveState()); event.accept()


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("LegumePhC Studio")
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    window = StudioWindow(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
