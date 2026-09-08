from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from .geometry_editor import (
    SHAPE_CHOICES, canonical_shape, editor_value, epsilon_from_editor,
    model_name, safe_number, shape_choice, uniaxial_matrix,
)
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
from ..motifs import triangular_motifs
from ..units import LENGTH_UNITS, frequency_factor

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


def _number_text(value: Any) -> str:
    return format(float(value), ".12g")


def _optional_numbers(text: str, *, count: int | None = None) -> list[float] | None:
    values = [safe_number(part) for part in text.split(",") if part.strip()]
    if not values:
        return None
    if count is not None and len(values) != count:
        raise ValueError(f"enter exactly {count} comma-separated values")
    return values


class StableComboBox(QtWidgets.QComboBox):
    """Non-native list popup used consistently throughout the workbench."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        view = QtWidgets.QListView(self)
        view.setUniformItemSizes(True)
        view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setView(view)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)


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

    COLUMNS = ("Name", "Shape", "Radius r/a", "Rotation (deg)", "Center x/a", "Center y/a", "Material", "Sides")

    def __init__(self, parent: QtWidgets.QWidget, case: dict[str, Any], representation: str = "n"):
        super().__init__(parent)
        self.setWindowTitle("Edit lattice sites")
        self.resize(900, 360)
        self.result: list[dict[str, Any]] | None = None
        self.case = case
        self.representation = representation
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeaderItem(6).setText("n" if representation == "n" else "ε")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for motif in _motifs(case):
            self._add_row(motif)
        note = QtWidgets.QLabel("Geometry parameters are defined before affine transformation. Each site may use a different shape, size, rotation and material.")
        note.setWordWrap(True)
        add = QtWidgets.QPushButton("Add site")
        remove = QtWidgets.QPushButton("Remove selected")
        center = QtWidgets.QPushButton("Center selected in unit cell")
        add.clicked.connect(self._add_site)
        remove.clicked.connect(self._remove)
        center.clicked.connect(lambda: self._center(case))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        tools = QtWidgets.QHBoxLayout()
        tools.addWidget(add); tools.addWidget(remove); tools.addWidget(center); tools.addStretch()
        layout = QtWidgets.QVBoxLayout(self)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #d32f2f")
        layout.addWidget(note); layout.addWidget(self.table); layout.addLayout(tools); layout.addWidget(self.error); layout.addWidget(buttons)

    def _add_row(self, motif: dict[str, Any]) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        shape = shape_choice(motif.get("kind", "circle"), motif.get("sides"))
        values = [motif.get("name", f"Site {row + 1}"), shape, motif.get("radius", 0.2), motif.get("angle_degrees", 0.0), *motif.get("center", [0.5, 0.0]), editor_value(float(motif.get("epsilon", 1.0)), self.representation), motif.get("sides", 6)]
        for column, value in enumerate(values):
            if column == 1:
                combo = StableComboBox(); combo.addItems(SHAPE_CHOICES); combo.setCurrentText(str(value)); combo.currentTextChanged.connect(lambda _text, r=row: self._shape_changed(r)); self.table.setCellWidget(row, column, combo)
            else:
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(_number_text(value) if isinstance(value, (float, np.floating)) else str(value)))
        self._shape_changed(row)

    def _shape_changed(self, row: int) -> None:
        if row >= self.table.rowCount():
            return
        shape = self.table.cellWidget(row, 1).currentText()
        angle = self.table.item(row, 3)
        sides = self.table.item(row, 7)
        angle.setFlags(angle.flags() | QtCore.Qt.ItemFlag.ItemIsEditable if shape != "Circle" else angle.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        sides.setFlags(sides.flags() | QtCore.Qt.ItemFlag.ItemIsEditable if shape == "Regular polygon" else sides.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        if shape in {"Triangle", "Square"}:
            sides.setText("3" if shape == "Triangle" else "4")
        elif shape == "Circle":
            angle.setText("0")
            sides.setText("0")

    def _row_motif(self, row: int) -> dict[str, Any]:
        field = "shape"
        try:
            shape = self.table.cellWidget(row, 1).currentText()
            field = "sides"
            raw_sides = safe_number(self.table.item(row, 7).text()) if shape == "Regular polygon" else self.table.item(row, 7).text()
            if shape == "Regular polygon" and not float(raw_sides).is_integer():
                raise ValueError("must be an integer")
            kind, sides = canonical_shape(shape, raw_sides)
            values = {}
            for column, name in ((2, "radius"), (3, "rotation"), (4, "center x"), (5, "center y"), (6, "n")):
                field = name
                values[name] = safe_number(self.table.item(row, column).text())
            if values["radius"] <= 0:
                field = "radius"; raise ValueError("must be positive")
            if values["n"] <= 0:
                field = "n"; raise ValueError("must be positive")
            return {
                "name": self.table.item(row, 0).text().strip() or f"Site {row + 1}",
                "kind": kind, "radius": values["radius"], "sides": sides,
                "angle_degrees": 0.0 if kind == "circle" else values["rotation"],
                "center": [values["center x"], values["center y"]],
                "epsilon": epsilon_from_editor(values["n"], self.representation),
            }
        except Exception as exc:
            raise ValueError(f"Site {row + 1}, {field}: {exc}") from exc

    def _add_site(self) -> None:
        new = {"name": f"Site {self.table.rowCount() + 1}", "kind": "circle", "radius": 0.2, "angle_degrees": 0.0, "center": [0.5, 0.0], "epsilon": 1.0, "sides": 0}
        if self.case.get("lattice") == "triangular" and self.table.rowCount() == 1:
            try:
                first = self._row_motif(0)
                scale = float(self.case.get("lattice_constant", 1.0))
                cell_center = np.asarray([0.5, 0.0]) * scale
                if np.allclose(first["center"], cell_center, rtol=0, atol=1e-12):
                    locations = triangular_motifs((first["radius"], first["radius"]), (first["angle_degrees"], first["angle_degrees"]), kind=first["kind"], sides=first["sides"] or 3, epsilon=first["epsilon"], scale=scale)
                    first["center"] = locations[0]["center"]
                    first["name"] = "Site A"
                    new = {**first, "name": "Site B", "center": locations[1]["center"]}
                    self.table.item(0, 0).setText(first["name"])
                    self.table.item(0, 4).setText(_number_text(first["center"][0]))
                    self.table.item(0, 5).setText(_number_text(first["center"][1]))
            except ValueError:
                pass
        self._add_row(new)

    def _remove(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if self.table.rowCount() - len(rows) < 1:
            self.error.setText("Keep at least one site")
            return
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
                motifs.append(self._row_motif(row))
            self.result = motifs
            self.accept()
        except Exception as exc:
            self.error.setText(str(exc))


class CentersDialog(QtWidgets.QDialog):
    """Structured editor for Berry sampling centres."""

    def __init__(self, parent: QtWidgets.QWidget, centers: list[list[float]]):
        super().__init__(parent); self.setWindowTitle("Explicit Berry centers"); self.resize(420, 320); self.result = None
        self.table = QtWidgets.QTableWidget(0, 2); self.table.setHorizontalHeaderLabels(["qₓ", "qᵧ"]); self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for center in centers: self._add(center)
        add = QtWidgets.QPushButton("Add center"); remove = QtWidgets.QPushButton("Remove selected")
        add.clicked.connect(lambda: self._add([0.0, 0.0])); remove.clicked.connect(self._remove)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject)
        self.error = QtWidgets.QLabel(); self.error.setStyleSheet("color: #d32f2f")
        tools = QtWidgets.QHBoxLayout(); tools.addWidget(add); tools.addWidget(remove); tools.addStretch()
        layout = QtWidgets.QVBoxLayout(self); layout.addWidget(QtWidgets.QLabel("Reduced Cartesian coordinates, k = 2πq/a")); layout.addWidget(self.table); layout.addLayout(tools); layout.addWidget(self.error); layout.addWidget(buttons)

    def _add(self, center: list[float]) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        for column, value in enumerate(center): self.table.setItem(row, column, QtWidgets.QTableWidgetItem(_number_text(value)))

    def _remove(self) -> None:
        for row in sorted({item.row() for item in self.table.selectedIndexes()}, reverse=True): self.table.removeRow(row)

    def _accept(self) -> None:
        try:
            if self.table.rowCount() < 1: raise ValueError("Add at least one center")
            self.result = [[safe_number(self.table.item(row, column).text()) for column in range(2)] for row in range(self.table.rowCount())]
            self.accept()
        except Exception as exc: self.error.setText(str(exc))


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
        self.last_worker_error: str | None = None
        self._cancelling = False
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
        self.view_menu = self.menuBar().addMenu("View")
        self.view_menu.addAction("Reset Plot View", lambda: self.result_canvas.plot.autoRange())
        self.view_menu.addSeparator()
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Geometry coordinates and affine transform", self._geometry_help)
        help_menu.addAction("Berry targets and sampling", self._berry_help)
        help_menu.addAction("Projects and parameter presets", self._project_help)

    def _show_help(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.information(self, title, text)

    def _geometry_help(self) -> None:
        self._show_help("Geometry coordinates", "Site centers, radii and rotations are Cartesian values in units of a before affine transformation. Standard triangular and square direct bases are automatic. Custom basis, full affine matrix and motif translation are under Advanced geometry. Numeric fields accept pi, sqrt(), root(), arithmetic and parentheses.")

    def _berry_help(self) -> None:
        self._show_help("Berry targets and sampling", "Single band computes the Abelian curvature of one band. Composite subspace computes the trace of the non-Abelian curvature for the complete contiguous band range; it is not the curvature of either individual band. qₓ/qᵧ are reduced Cartesian coordinates with k=2πq/a. Berry step controls the Wilson loop, while grid size controls sampling centers.")

    def _project_help(self) -> None:
        self._show_help("Projects and presets", "A Project stores the model, calculation nodes, display settings and immutable result references. A Parameter Preset stores reusable model and calculation parameters only; it contains no results.")

    def result_canvas_clear_later(self) -> None:
        if hasattr(self, "result_canvas"): self.result_canvas.clear_pins()

    def _tree_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Project", self); dock.setObjectName("ProjectDock")
        self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderHidden(True); self.tree.currentItemChanged.connect(self._tree_selected)
        dock.setWidget(self.tree); self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock); self.view_menu.addAction(dock.toggleViewAction())

    def _settings_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Settings", self); dock.setObjectName("SettingsDock")
        self.settings_tabs = QtWidgets.QTabWidget(); dock.setWidget(self.settings_tabs)
        self.model_page = QtWidgets.QWidget(); self.calc_page = QtWidgets.QWidget(); self.plot_page = QtWidgets.QWidget()
        self.settings_tabs.addTab(self.model_page, "Geometry"); self.settings_tabs.addTab(self.calc_page, "Calculation"); self.settings_tabs.addTab(self.plot_page, "Plot Style")
        self._build_model_form(); self._build_calc_form(); self._build_plot_form()
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock); self.view_menu.addAction(dock.toggleViewAction())

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
        dock.setWidget(tabs); self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock); self.view_menu.addAction(dock.toggleViewAction())

    def result_canvas_select_later(self, index: int) -> None:
        if hasattr(self, "result_canvas"): self.result_canvas.select_row(index)

    def _build_model_form(self) -> None:
        form = QtWidgets.QFormLayout(self.model_page)
        self.lattice = StableComboBox(); self.lattice.addItems(["triangular", "square", "custom"])
        self.actual_a = QtWidgets.QLineEdit(); self.length_unit = StableComboBox(); self.length_unit.addItems(["nm", "μm", "m"])
        length_row = QtWidgets.QWidget(); length_layout = QtWidgets.QHBoxLayout(length_row); length_layout.setContentsMargins(0, 0, 0, 0); length_layout.addWidget(self.actual_a); length_layout.addWidget(self.length_unit)
        self.material_rep = StableComboBox(); self.material_rep.addItems(["n", "epsilon"])
        self.background_material = QtWidgets.QLineEdit()
        self.basis_summary = QtWidgets.QLabel(); self.basis_summary.setWordWrap(True)
        self.basis_policy = StableComboBox(); self.basis_policy.addItems(["auto", "native"])
        self.deformation = StableComboBox(); self.deformation.addItems(["none", "uniaxial", "custom"])
        self.factor = QtWidgets.QLineEdit(); self.angle = QtWidgets.QLineEdit()
        self.sites_button = QtWidgets.QPushButton("Edit sites…"); self.sites_button.clicked.connect(self.edit_sites)
        form.addRow("Lattice", self.lattice); form.addRow("Actual lattice constant", length_row); form.addRow("Material representation", self.material_rep); form.addRow("Background material", self.background_material); form.addRow("Direct basis", self.basis_summary); form.addRow("Basis policy", self.basis_policy); form.addRow("Sites", self.sites_button); form.addRow("Deformation", self.deformation); form.addRow("Stretch factor", self.factor); form.addRow("Stretch angle (deg)", self.angle)
        self.advanced = QtWidgets.QGroupBox("Advanced geometry"); self.advanced.setCheckable(True)
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced); self.advanced_contents = QtWidgets.QWidget(); advanced_layout.addWidget(self.advanced_contents)
        advanced_form = QtWidgets.QFormLayout(self.advanced_contents)
        self.basis_edits = [QtWidgets.QLineEdit() for _ in range(4)]
        basis_box = QtWidgets.QWidget(); basis_grid = QtWidgets.QGridLayout(basis_box); basis_grid.setContentsMargins(0, 0, 0, 0)
        for index, edit in enumerate(self.basis_edits): basis_grid.addWidget(edit, index // 2, index % 2)
        self.affine_edits = [QtWidgets.QLineEdit() for _ in range(4)]
        affine_box = QtWidgets.QWidget(); affine_grid = QtWidgets.QGridLayout(affine_box); affine_grid.setContentsMargins(0, 0, 0, 0)
        for index, edit in enumerate(self.affine_edits): affine_grid.addWidget(edit, index // 2, index % 2)
        self.tx = QtWidgets.QLineEdit(); self.ty = QtWidgets.QLineEdit(); translation_box = QtWidgets.QWidget(); translation_layout = QtWidgets.QHBoxLayout(translation_box); translation_layout.setContentsMargins(0, 0, 0, 0); translation_layout.addWidget(self.tx); translation_layout.addWidget(self.ty)
        self.geometry_scale = QtWidgets.QLineEdit()
        advanced_form.addRow("Custom direct basis", basis_box); advanced_form.addRow("Affine matrix", affine_box); advanced_form.addRow("Motif translation x/y", translation_box); advanced_form.addRow("Geometry scale", self.geometry_scale); self.advanced_contents.setVisible(False)
        form.addRow(self.advanced)
        self.lattice.currentTextChanged.connect(self._lattice_changed)
        self.material_rep.currentTextChanged.connect(self._material_rep_changed)
        self.length_unit.currentTextChanged.connect(self._length_unit_changed)
        self.deformation.currentTextChanged.connect(self._deformation_changed)
        for combo in (self.basis_policy,): combo.currentTextChanged.connect(self._model_changed)
        for edit in (self.actual_a, self.background_material, self.factor, self.angle, *self.basis_edits, *self.affine_edits, self.tx, self.ty, self.geometry_scale): edit.editingFinished.connect(self._model_changed)
        self.advanced.toggled.connect(self.advanced_contents.setVisible); self.advanced.toggled.connect(self._advanced_changed)

    def _lattice_changed(self, *_args) -> None:
        lattice = self.lattice.currentText()
        if lattice == "triangular":
            self.basis_summary.setText("a₁=(a/2, √3a/2), a₂=(a/2, −√3a/2)")
        elif lattice == "square":
            self.basis_summary.setText("a₁=(a, 0), a₂=(0, a)")
        else:
            self.basis_summary.setText("Custom dimensionless direct basis")
        for edit in self.basis_edits: edit.setEnabled(lattice == "custom")
        self._model_changed()

    def _deformation_changed(self, *_args) -> None:
        kind = self.deformation.currentText()
        self.factor.setEnabled(kind == "uniaxial"); self.angle.setEnabled(kind == "uniaxial")
        for edit in self.affine_edits: edit.setEnabled(kind == "custom")
        self._model_changed()

    def _advanced_changed(self, expanded: bool) -> None:
        if self._building: return
        self.project.setdefault("ui_state", {})["advanced_expanded"] = bool(expanded); self.dirty = True

    def _material_rep_changed(self, representation: str) -> None:
        if self._building: return
        epsilon = float(self.project["model"]["geometry"].get("epsilon_background", 7.29))
        self.background_material.setText(_number_text(editor_value(epsilon, representation)))
        self.project.setdefault("ui_state", {})["material_representation"] = representation
        self.dirty = True; self.statusBar().showMessage("Material display changed — scientific epsilon unchanged")

    def _length_unit_changed(self, unit: str) -> None:
        if self._building: return
        length = self.project["model"].get("actual_lattice_constant_m")
        self.actual_a.setText("" if length is None else _number_text(float(length) / LENGTH_UNITS[unit]))
        self.project.setdefault("ui_state", {})["length_unit"] = unit
        self.dirty = True

    def _build_calc_form(self) -> None:
        form = QtWidgets.QFormLayout(self.calc_page); self.calc_form = form; self._calc_rows = {}
        self.operation = StableComboBox(); [self.operation.addItem(label, key) for key, label in OPERATIONS.items()]
        self.qx = QtWidgets.QDoubleSpinBox(); self.qy = QtWidgets.QDoubleSpinBox()
        for widget in (self.qx, self.qy): widget.setRange(-100, 100); widget.setDecimals(9)
        self.band = QtWidgets.QSpinBox(); self.band.setRange(1, 100)
        self.first_band = QtWidgets.QSpinBox(); self.first_band.setRange(1, 100)
        self.last_band = QtWidgets.QSpinBox(); self.last_band.setRange(1, 100)
        self.gmax = QtWidgets.QDoubleSpinBox(); self.gmax.setRange(0.01, 100); self.gmax.setDecimals(6)
        self.numeig = QtWidgets.QSpinBox(); self.numeig.setRange(1, 200)
        self.pol = StableComboBox(); self.pol.addItems(["te", "tm"])
        self.path = StableComboBox(); self.path.addItem("Automatic high-symmetry path", "identity")
        self.berry_target = StableComboBox(); self.berry_target.addItem("Single band", "single_band"); self.berry_target.addItem("Composite subspace", "composite_subspace")
        self.berry_sampling = StableComboBox(); self.berry_sampling.addItem("Single plaquette", "single_plaquette"); self.berry_sampling.addItem("First BZ grid", "first_bz_grid"); self.berry_sampling.addItem("Explicit centers", "explicit_centers")
        self.samples = QtWidgets.QSpinBox(); self.samples.setRange(2, 10000)
        self.grid_size = QtWidgets.QSpinBox(); self.grid_size.setRange(2, 1000)
        self.field_grid = QtWidgets.QSpinBox(); self.field_grid.setRange(2, 1000)
        self.efs_grid = QtWidgets.QSpinBox(); self.efs_grid.setRange(2, 1000)
        self.berry_step = QtWidgets.QDoubleSpinBox(); self.berry_step.setRange(1e-8, 1); self.berry_step.setDecimals(9)
        self.centers_button = QtWidgets.QPushButton("Edit explicit centers…"); self.centers_button.clicked.connect(self.edit_centers)
        self.source_result = StableComboBox(); self.frequency_window = QtWidgets.QLineEdit(); self.frequency_samples = QtWidgets.QLineEdit(); self.response_weights = QtWidgets.QLineEdit(); self.occupation = QtWidgets.QLineEdit()
        for key, label, widget in (("operation", "Calculation", self.operation), ("qx", "qₓ (reduced Cartesian)", self.qx), ("qy", "qᵧ (reduced Cartesian)", self.qy), ("band", "Target band (one-based)", self.band), ("first_band", "First composite band", self.first_band), ("last_band", "Last composite band", self.last_band), ("gmax", "gmax", self.gmax), ("numeig", "Eigenmodes", self.numeig), ("pol", "Polarization", self.pol), ("path", "Band path", self.path), ("samples", "Samples per segment", self.samples), ("field_grid", "Field grid size", self.field_grid), ("berry_grid", "Berry grid size", self.grid_size), ("efs_grid", "EFS grid size", self.efs_grid), ("berry_step", "Berry step", self.berry_step), ("berry_target", "Berry target", self.berry_target), ("berry_sampling", "Berry sampling", self.berry_sampling), ("centers", "Sampling centers", self.centers_button), ("source", "Qualified Berry source", self.source_result), ("frequency_window", "Frequency window", self.frequency_window), ("frequency_samples", "Frequency samples", self.frequency_samples), ("response_weights", "Response weights", self.response_weights), ("occupation", "Occupation", self.occupation)):
            form.addRow(label, widget); self._calc_rows[key] = widget
        calculation_tools = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run"); self.cancel_button = QtWidgets.QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.run_button.clicked.connect(self.run); self.cancel_button.clicked.connect(self.cancel); calculation_tools.addWidget(self.run_button); calculation_tools.addWidget(self.cancel_button)
        for text, slot in (("Add", self.add_calculation), ("Copy", self.copy_calculation), ("Rename", self.rename_calculation), ("Delete", self.delete_calculation)):
            button = QtWidgets.QPushButton(text); button.clicked.connect(slot); calculation_tools.addWidget(button)
        form.addRow(calculation_tools)
        self.plot_after = QtWidgets.QCheckBox("Plot after Run"); self.plot_after.setChecked(True); form.addRow(self.plot_after)
        self.operation.currentIndexChanged.connect(self._calculation_changed)
        self.berry_target.currentIndexChanged.connect(self._berry_controls_changed); self.berry_sampling.currentIndexChanged.connect(self._berry_controls_changed)
        for widget in (self.qx, self.qy, self.band, self.first_band, self.last_band, self.gmax, self.numeig, self.pol, self.path, self.samples, self.field_grid, self.grid_size, self.efs_grid, self.berry_step, self.source_result, self.frequency_window, self.frequency_samples, self.response_weights, self.occupation, self.plot_after):
            if isinstance(widget, QtWidgets.QComboBox): signal = widget.currentTextChanged
            elif isinstance(widget, QtWidgets.QLineEdit): signal = widget.editingFinished
            elif isinstance(widget, QtWidgets.QCheckBox): signal = widget.toggled
            else: signal = widget.valueChanged
            signal.connect(self._calculation_changed)

    def _set_calc_visible(self, key: str, visible: bool) -> None:
        widget = self._calc_rows[key]
        self.calc_form.setRowVisible(widget, visible)

    def _update_calculation_visibility(self) -> None:
        operation = self.operation.currentData() or "frequency_at_k"
        visible = {"operation", "gmax", "numeig", "pol"}
        visible.update({
            "frequency_at_k": {"qx", "qy", "band"},
            "band_structure": {"path", "samples"},
            "fields_energy": {"qx", "qy", "first_band", "last_band", "field_grid"},
            "efs": {"first_band", "last_band", "efs_grid"},
            "berry": {"berry_target", "berry_sampling", "berry_step"},
            "berry_curvature_dipole": {"source", "frequency_window", "frequency_samples", "response_weights", "occupation"},
        }[operation])
        if operation == "berry":
            if self.berry_target.currentData() == "single_band": visible.add("band")
            else: visible.update({"first_band", "last_band"})
            sampling = self.berry_sampling.currentData()
            if sampling == "single_plaquette": visible.update({"qx", "qy"})
            elif sampling == "first_bz_grid": visible.add("berry_grid")
            else: visible.add("centers")
        for key in self._calc_rows: self._set_calc_visible(key, key in visible)
        self.numeig.setReadOnly(operation == "berry")
        self._update_plot_visibility(operation)

    def _berry_controls_changed(self, *_args) -> None:
        if self._building: return
        target = self.band.value() if self.berry_target.currentData() == "single_band" else self.last_band.value()
        self.numeig.setValue(target + 1)
        self._update_calculation_visibility(); self._calculation_changed()

    def edit_centers(self) -> None:
        calculation = self._selected_calculation()["parameters"]
        dialog = CentersDialog(self, list(calculation.get("centers", [])))
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.result is not None:
            calculation["centers"] = dialog.result; self.dirty = True; self.statusBar().showMessage("Explicit Berry centers updated")

    def _build_plot_form(self) -> None:
        form = QtWidgets.QFormLayout(self.plot_page); self.plot_form = form; self._plot_rows = {}
        self.unit = StableComboBox(); self.unit.addItems(["Normalized", "GHz", "THz"])
        self.band_lines = QtWidgets.QCheckBox(); self.band_markers = QtWidgets.QCheckBox(); self.grid = QtWidgets.QCheckBox(); self.legend = QtWidgets.QCheckBox(); self.colorbar = QtWidgets.QCheckBox(); self.sample_centers = QtWidgets.QCheckBox()
        self.render_mode = StableComboBox(); self.render_mode.addItem("Sample-cell tiling", "sample_cells"); self.render_mode.addItem("Linear interpolation", "linear_interpolation")
        self.cmap = StableComboBox(); self.cmap.setEditable(True); self.cmap.addItems(["RdBu_r", "viridis", "plasma", "magma", "coolwarm"])
        self.width = QtWidgets.QSpinBox(); self.width.setRange(100, 10000); self.height = QtWidgets.QSpinBox(); self.height.setRange(100, 10000); self.dpi = QtWidgets.QSpinBox(); self.dpi.setRange(30, 1200)
        self.title_edit = QtWidgets.QLineEdit(); self.x_label_edit = QtWidgets.QLineEdit(); self.y_label_edit = QtWidgets.QLineEdit()
        self.xmin = QtWidgets.QLineEdit(); self.xmax = QtWidgets.QLineEdit(); self.ymin = QtWidgets.QLineEdit(); self.ymax = QtWidgets.QLineEdit()
        self.linewidth = QtWidgets.QDoubleSpinBox(); self.linewidth.setRange(.01, 100); self.linewidth.setDecimals(3); self.marker_size = QtWidgets.QDoubleSpinBox(); self.marker_size.setRange(.01, 100); self.marker_size.setDecimals(3)
        self.component = QtWidgets.QSpinBox(); self.component.setRange(0, 1000); self.field_quantity = StableComboBox(); self.field_quantity.addItems(["energy_density", "E2", "H2"])
        self.vmin = QtWidgets.QLineEdit(); self.vmax = QtWidgets.QLineEdit(); self.interpolation_resolution = QtWidgets.QSpinBox(); self.interpolation_resolution.setRange(8, 2000); self.efs_levels = QtWidgets.QLineEdit()
        def pair(left, right):
            host = QtWidgets.QWidget(); layout = QtWidgets.QHBoxLayout(host); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(left); layout.addWidget(right); return host
        for key, label, widget in (("unit", "Frequency unit", self.unit), ("width", "Export width (px)", self.width), ("height", "Export height (px)", self.height), ("dpi", "DPI", self.dpi), ("title", "Title", self.title_edit), ("xlabel", "X label", self.x_label_edit), ("ylabel", "Y label", self.y_label_edit), ("xlimits", "X min / max", pair(self.xmin, self.xmax)), ("ylimits", "Y min / max", pair(self.ymin, self.ymax)), ("lines", "Band lines", self.band_lines), ("markers", "Band markers", self.band_markers), ("linewidth", "Line width", self.linewidth), ("marker_size", "Marker size", self.marker_size), ("grid", "Grid", self.grid), ("legend", "Legend", self.legend), ("field_quantity", "Field quantity", self.field_quantity), ("component", "Component / band", self.component), ("render", "Berry render", self.render_mode), ("resolution", "Interpolation resolution", self.interpolation_resolution), ("cmap", "Colormap", self.cmap), ("vmin", "Color minimum", self.vmin), ("vmax", "Color maximum", self.vmax), ("colorbar", "Colorbar", self.colorbar), ("centers", "Sample centers", self.sample_centers), ("efs_levels", "EFS levels", self.efs_levels)):
            form.addRow(label, widget); self._plot_rows[key] = widget
        apply = QtWidgets.QPushButton("Apply to Current Plot"); apply.clicked.connect(self.plot_selected); form.addRow(apply)
        for widget in (self.unit, self.band_lines, self.band_markers, self.grid, self.legend, self.render_mode, self.cmap, self.colorbar, self.sample_centers, self.width, self.height, self.dpi, self.title_edit, self.x_label_edit, self.y_label_edit, self.xmin, self.xmax, self.ymin, self.ymax, self.linewidth, self.marker_size, self.component, self.field_quantity, self.vmin, self.vmax, self.interpolation_resolution, self.efs_levels):
            if widget is self.unit: continue
            if isinstance(widget, QtWidgets.QComboBox): signal = widget.currentTextChanged
            elif isinstance(widget, QtWidgets.QCheckBox): signal = widget.toggled
            elif isinstance(widget, QtWidgets.QLineEdit): signal = widget.editingFinished
            else: signal = widget.valueChanged
            signal.connect(self._plot_changed)
        self.unit.currentTextChanged.connect(self._frequency_unit_changed)

    def _update_plot_visibility(self, operation: str | None = None) -> None:
        operation = operation or (self.current_view.operation if self.current_view else self.operation.currentData()) or "frequency_at_k"
        visible = {"width", "height", "dpi", "title", "xlabel", "ylabel", "xlimits", "ylimits", "grid", "legend"}
        specific = {
            "band_structure": {"unit", "lines", "markers", "linewidth", "marker_size"},
            "frequency_at_k": {"unit"},
            "berry": {"render", "resolution", "cmap", "vmin", "vmax", "colorbar", "centers"},
            "efs": {"unit", "component", "cmap", "colorbar", "efs_levels"},
            "fields_energy": {"field_quantity", "component", "cmap", "colorbar", "vmin", "vmax"},
            "berry_curvature_dipole": {"component", "cmap", "colorbar"},
        }.get(operation, set())
        visible |= specific
        for key, widget in self._plot_rows.items(): self.plot_form.setRowVisible(widget, key in visible)
        self.interpolation_resolution.setEnabled(self.render_mode.currentData() == "linear_interpolation")

    def _load_plot_controls(self, style: dict[str, Any]) -> None:
        widgets = [self.unit, self.band_lines, self.band_markers, self.grid, self.legend, self.colorbar, self.sample_centers, self.render_mode, self.cmap, self.width, self.height, self.dpi, self.title_edit, self.x_label_edit, self.y_label_edit, self.xmin, self.xmax, self.ymin, self.ymax, self.linewidth, self.marker_size, self.component, self.field_quantity, self.vmin, self.vmax, self.interpolation_resolution, self.efs_levels]
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        try:
            self.unit.setCurrentText(style.get("frequency_unit", "Normalized")); self.band_lines.setChecked(bool(style.get("band_line", True))); self.band_markers.setChecked(bool(style.get("band_markers", False))); self.grid.setChecked(bool(style.get("grid", True))); self.legend.setChecked(bool(style.get("legend", True))); self.colorbar.setChecked(bool(style.get("colorbar", True))); self.sample_centers.setChecked(bool(style.get("show_sample_centers", False))); self.render_mode.setCurrentIndex(max(0, self.render_mode.findData(style.get("berry_render_mode", "sample_cells")))); self.cmap.setCurrentText(style.get("cmap", "RdBu_r")); self.width.setValue(int(style.get("width_px", 900))); self.height.setValue(int(style.get("height_px", 600))); self.dpi.setValue(int(style.get("dpi", 100)))
            self.title_edit.setText(str(style.get("title", ""))); self.x_label_edit.setText(str(style.get("x_label", ""))); self.y_label_edit.setText(str(style.get("y_label", "")))
            xlimits, ylimits = style.get("x_limits"), style.get("y_limits"); self.xmin.setText("" if not xlimits else _number_text(xlimits[0])); self.xmax.setText("" if not xlimits else _number_text(xlimits[1])); self.ymin.setText("" if not ylimits else _number_text(ylimits[0])); self.ymax.setText("" if not ylimits else _number_text(ylimits[1]))
            self.linewidth.setValue(float(style.get("linewidth", 1.5))); self.marker_size.setValue(float(style.get("marker_size", 4))); self.component.setValue(int(style.get("component_index", 0))); self.field_quantity.setCurrentText(style.get("field_quantity", "energy_density")); self.vmin.setText("" if style.get("berry_vmin") is None else _number_text(style["berry_vmin"])); self.vmax.setText("" if style.get("berry_vmax") is None else _number_text(style["berry_vmax"])); self.interpolation_resolution.setValue(int(style.get("interpolation_resolution", 160))); self.efs_levels.setText(", ".join(_number_text(value) for value in style.get("efs_levels") or []))
        finally:
            del blockers

    def _frequency_unit_changed(self, *_args) -> None:
        if self._building: return
        old = self.project["plot"].get("frequency_unit", "Normalized"); new = self.unit.currentText()
        try:
            length = self.project["model"].get("actual_lattice_constant_m")
            if self.current_view is not None:
                saved = self.current_view.config.get("config", self.current_view.config); length = saved.get("case", {}).get("actual_lattice_constant_m", saved.get("actual_lattice_constant_m"))
            ratio = frequency_factor(new, length) / frequency_factor(old, length)
            operation = self.current_view.operation if self.current_view is not None else self.operation.currentData()
            if operation == "efs" and self.efs_levels.text().strip():
                self.efs_levels.setText(", ".join(_number_text(value * ratio) for value in _optional_numbers(self.efs_levels.text()) or []))
            if operation in {"band_structure", "frequency_at_k"} and self.ymin.text().strip() and self.ymax.text().strip():
                self.ymin.setText(_number_text(safe_number(self.ymin.text()) * ratio)); self.ymax.setText(_number_text(safe_number(self.ymax.text()) * ratio))
            self._sync_plot()
            if self._selected_result_entry() is not None: self.plot_selected()
        except ValueError as exc:
            blocker = QtCore.QSignalBlocker(self.unit); self.unit.setCurrentText(old); del blocker; self.statusBar().showMessage(str(exc))

    def populate(self) -> None:
        self._building = True
        model = self.project["model"]; geometry = model["geometry"]; deformation = model.get("deformation", {})
        state = self.project.setdefault("ui_state", {})
        length_unit = state.get("length_unit", "nm"); representation = state.get("material_representation", "n")
        self.lattice.setCurrentText(model.get("lattice", "triangular")); self.length_unit.setCurrentText(length_unit); self.actual_a.setText("" if model.get("actual_lattice_constant_m") is None else _number_text(float(model["actual_lattice_constant_m"]) / LENGTH_UNITS[length_unit])); self.material_rep.setCurrentText(representation); self.background_material.setText(_number_text(editor_value(float(geometry.get("epsilon_background", 7.29)), representation))); self.basis_policy.setCurrentText(model.get("basis_policy", "auto")); self.deformation.setCurrentText(deformation.get("kind", "none")); self.factor.setText(_number_text(deformation.get("factor", 1))); self.angle.setText(_number_text(deformation.get("angle_degrees", 0)))
        basis = np.asarray(model.get("direct_basis", [[.5, .5], [np.sqrt(3)/2, -np.sqrt(3)/2]]), dtype=float).reshape(-1)
        affine = np.asarray(deformation.get("linear", model.get("affine", {}).get("linear", np.eye(2))), dtype=float).reshape(-1)
        translation = deformation.get("translation", model.get("affine", {}).get("translation", [0, 0]))
        for edit, value in zip(self.basis_edits, basis): edit.setText(_number_text(value))
        for edit, value in zip(self.affine_edits, affine): edit.setText(_number_text(value))
        self.tx.setText(_number_text(translation[0])); self.ty.setText(_number_text(translation[1])); self.geometry_scale.setText(_number_text(model.get("lattice_constant", 1)))
        self.advanced.setChecked(bool(state.get("advanced_expanded", False)))
        self._lattice_changed(); self._deformation_changed()
        self._load_calculation_controls()
        selected_result = self._selected_result_entry(); style = {**self.project["plot"], **(selected_result.get("plot", {}) if selected_result else {})}; self.unit.setCurrentText(style.get("frequency_unit", "Normalized")); self.band_lines.setChecked(bool(style.get("band_line", True))); self.band_markers.setChecked(bool(style.get("band_markers", False))); self.grid.setChecked(bool(style.get("grid", True))); self.legend.setChecked(bool(style.get("legend", True))); self.colorbar.setChecked(bool(style.get("colorbar", True))); self.sample_centers.setChecked(bool(style.get("show_sample_centers", False))); self.render_mode.setCurrentIndex(max(0, self.render_mode.findData(style.get("berry_render_mode", "sample_cells")))); self.cmap.setCurrentText(style.get("cmap", "RdBu_r")); self.width.setValue(int(style.get("width_px", 900))); self.height.setValue(int(style.get("height_px", 600))); self.dpi.setValue(int(style.get("dpi", 100)))
        self.title_edit.setText(str(style.get("title", ""))); self.x_label_edit.setText(str(style.get("x_label", ""))); self.y_label_edit.setText(str(style.get("y_label", "")))
        xlimits, ylimits = style.get("x_limits"), style.get("y_limits")
        self.xmin.setText("" if not xlimits else _number_text(xlimits[0])); self.xmax.setText("" if not xlimits else _number_text(xlimits[1])); self.ymin.setText("" if not ylimits else _number_text(ylimits[0])); self.ymax.setText("" if not ylimits else _number_text(ylimits[1]))
        self.linewidth.setValue(float(style.get("linewidth", 1.5))); self.marker_size.setValue(float(style.get("marker_size", 4))); self.component.setValue(int(style.get("component_index", 0))); self.field_quantity.setCurrentText(style.get("field_quantity", "energy_density")); self.vmin.setText("" if style.get("berry_vmin") is None else _number_text(style["berry_vmin"])); self.vmax.setText("" if style.get("berry_vmax") is None else _number_text(style["berry_vmax"])); self.interpolation_resolution.setValue(int(style.get("interpolation_resolution", 160))); self.efs_levels.setText(", ".join(_number_text(value) for value in style.get("efs_levels") or []))
        self._building = False
        self._populate_tree(); self._update_plot_visibility(); self.refresh_geometry(); self._update_title()

    def _selected_calculation(self) -> dict[str, Any]:
        selected = self.project.get("selected_node", {}).get("id")
        return next((item for item in self.project["calculations"] if item["id"] == selected), self.project["calculations"][0])

    def _load_calculation_controls(self) -> None:
        calculation = self._selected_calculation()["parameters"]
        widgets = [self.operation, self.qx, self.qy, self.band, self.first_band, self.last_band, self.gmax, self.numeig, self.pol, self.path, self.samples, self.field_grid, self.grid_size, self.efs_grid, self.berry_step, self.berry_target, self.berry_sampling, self.source_result, self.frequency_window, self.frequency_samples, self.response_weights, self.occupation, self.plot_after]
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        try:
            index = self.operation.findData(calculation.get("operation")); self.operation.setCurrentIndex(max(0, index))
            qpoint = calculation.get("qpoint", [0, 0]); self.qx.setValue(float(qpoint[0])); self.qy.setValue(float(qpoint[1]))
            self.band.setValue(int(calculation.get("band_one_based", 2))); self.first_band.setValue(int(calculation.get("berry_first_band", 2))); self.last_band.setValue(int(calculation.get("berry_last_band", 3)))
            self.gmax.setValue(float(calculation.get("gmax", 2))); self.numeig.setValue(int(calculation.get("numeig", 3))); self.pol.setCurrentText(calculation.get("polarization", "te")); self.path.setCurrentIndex(max(0, self.path.findData(calculation.get("path", "identity"))))
            self.samples.setValue(int(calculation.get("samples_per_segment", 16))); self.field_grid.setValue(int(calculation.get("grid_size", 8))); self.grid_size.setValue(int(calculation.get("grid_size", 8))); self.efs_grid.setValue(int(calculation.get("efs_grid_size", 5))); self.berry_step.setValue(float(calculation.get("berry_step", .02)))
            self.berry_target.setCurrentIndex(max(0, self.berry_target.findData(calculation.get("berry_target_mode", "single_band")))); self.berry_sampling.setCurrentIndex(max(0, self.berry_sampling.findData(calculation.get("sampling_mode", "single_plaquette"))))
            self._populate_berry_sources(calculation.get("berry_record_path"))
            self.frequency_window.setText(", ".join(_number_text(value) for value in calculation.get("frequency_window") or [])); self.frequency_samples.setText(", ".join(_number_text(value) for value in calculation.get("frequency_samples") or [])); self.response_weights.setText(", ".join(_number_text(value) for value in calculation.get("response_weights") or [])); self.occupation.setText(", ".join(_number_text(value) for value in calculation.get("occupation") or [])); self.plot_after.setChecked(bool(calculation.get("plot_after", True)))
        finally:
            del blockers
        self._update_calculation_visibility()

    def _populate_berry_sources(self, selected: str | None = None) -> None:
        blocker = QtCore.QSignalBlocker(self.source_result); self.source_result.clear(); self.source_result.addItem("No source selected", None)
        for result in reversed(self.project.get("results", [])):
            if result.get("calculation_snapshot", {}).get("operation") != "berry": continue
            reference = result.get("record_reference", {}); path = reference.get("path")
            label = self._result_label(result, available=True)
            self.source_result.addItem(label, path)
        index = self.source_result.findData(selected); self.source_result.setCurrentIndex(max(0, index)); del blocker

    def _populate_tree(self) -> None:
        selected = self.project.get("selected_node", {}); blocker = QtCore.QSignalBlocker(self.tree)
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
            item = QtWidgets.QTreeWidgetItem([self._result_label(result)]); item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("result", result["id"])); group.addChild(item)
        self.tree.expandAll()
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value(); data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if data == (selected.get("kind"), selected.get("id")): self.tree.setCurrentItem(item); break
            iterator += 1
        del blocker

    def _result_label(self, result: dict[str, Any], available: bool | None = None) -> str:
        calculation = result.get("calculation_snapshot", {}); operation = calculation.get("operation", "result")
        reference = result.get("record_reference", {}); stamp = str(reference.get("created_at") or Path(reference.get("path", "result")).name[:19]).replace("T", " ")
        geometry = result.get("model_snapshot", {}).get("name") or result.get("model_snapshot", {}).get("geometry", {}).get("name", "Geometry")
        if operation == "berry":
            target = f"band {calculation.get('band_one_based', '?')}" if calculation.get("berry_target_mode") == "single_band" else f"bands {calculation.get('berry_first_band', '?')}–{calculation.get('berry_last_band', '?')}"
            detail = f"{target}, grid={calculation.get('grid_size', '?')}, gmax={calculation.get('gmax', '?')}"
        elif operation == "band_structure": detail = f"gmax={calculation.get('gmax', '?')}, samples={calculation.get('samples_per_segment', '?')}"
        elif operation == "efs": detail = f"bands={calculation.get('composite_bands_one_based', '?')}, grid={calculation.get('efs_grid_size', '?')}"
        elif operation == "fields_energy": detail = f"bands={calculation.get('composite_bands_one_based', '?')}, grid={calculation.get('grid_size', '?')}"
        elif operation == "frequency_at_k": detail = f"band={calculation.get('band_one_based', '?')}, q={calculation.get('qpoint', '?')}"
        else: detail = "geometric" if not calculation.get("response_weights") and not calculation.get("occupation") else "physical weighting"
        status = "unavailable" if available is False else "available"
        return f"{stamp} · {geometry} · {detail} · {status}"

    def _tree_selected(self, current, _previous) -> None:
        if current is None or current.data(0, QtCore.Qt.ItemDataRole.UserRole) is None: return
        kind, identity = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self.project["selected_node"] = {"kind": kind, "id": identity}
        if kind == "geometry": self.settings_tabs.setCurrentWidget(self.model_page); self.central_tabs.setCurrentWidget(self.geometry_tabs)
        elif kind == "calculation":
            self.settings_tabs.setCurrentWidget(self.calc_page)
            self._load_calculation_controls()
        else:
            result = next(item for item in self.project["results"] if item["id"] == identity); self.project["selected_result"] = result["record_reference"]["path"]; self._load_plot_controls({**self.project["plot"], **result.get("plot", {})}); self.plot_selected(); self.central_tabs.setCurrentWidget(self.result_canvas)

    def _sync_model(self) -> None:
        if self._building: return
        model = self.project["model"]; geometry = model["geometry"]
        lattice = self.lattice.currentText(); scale = safe_number(self.geometry_scale.text())
        if scale <= 0: raise ValueError("Geometry scale must be positive")
        if lattice == "triangular": direct = [[.5, .5], [np.sqrt(3)/2, -np.sqrt(3)/2]]
        elif lattice == "square": direct = [[1, 0], [0, 1]]
        else: direct = np.asarray([safe_number(edit.text()) for edit in self.basis_edits]).reshape(2, 2).tolist()
        kind = self.deformation.currentText(); factor = safe_number(self.factor.text()); angle = safe_number(self.angle.text())
        if kind == "none": linear = np.eye(2)
        elif kind == "uniaxial": linear = uniaxial_matrix(factor, angle)
        else: linear = np.asarray([safe_number(edit.text()) for edit in self.affine_edits]).reshape(2, 2)
        if abs(float(np.linalg.det(linear))) < 1e-14: raise ValueError("Affine matrix must be non-singular")
        translation = [safe_number(self.tx.text()), safe_number(self.ty.text())]
        actual_text = self.actual_a.text().strip()
        actual_length = None if not actual_text else safe_number(actual_text) * LENGTH_UNITS[self.length_unit.currentText()]
        if actual_length is not None and actual_length <= 0: raise ValueError("Actual lattice constant must be positive")
        model.update({"lattice": lattice, "lattice_constant": scale, "actual_lattice_constant_m": actual_length, "direct_basis": direct, "basis_policy": self.basis_policy.currentText()})
        geometry["epsilon_background"] = epsilon_from_editor(self.background_material.text(), self.material_rep.currentText())
        deformation = {"kind": kind, "factor": factor, "angle_degrees": angle, "linear": linear.tolist(), "translation": translation}
        model["deformation"] = deformation; model["affine"] = {"linear": linear.tolist(), "translation": translation}
        model["name"] = model_name(lattice, _motifs(model)); geometry["name"] = model["name"]

    def _sync_calculation(self) -> None:
        calc = self._selected_calculation(); value = calc["parameters"]; operation = self.operation.currentData()
        target_mode = self.berry_target.currentData() or "single_band"
        bands = [self.band.value()] if operation == "berry" and target_mode == "single_band" else list(range(self.first_band.value(), self.last_band.value() + 1))
        numeig = max(bands) + 1 if operation == "berry" else self.numeig.value()
        value.update({"operation": operation, "qpoint": [self.qx.value(), self.qy.value()], "band_one_based": self.band.value(), "berry_first_band": self.first_band.value(), "berry_last_band": self.last_band.value(), "gmax": self.gmax.value(), "numeig": numeig, "polarization": self.pol.currentText(), "path": self.path.currentData() or "identity", "samples_per_segment": self.samples.value(), "grid_size": self.field_grid.value() if operation == "fields_energy" else self.grid_size.value(), "efs_grid_size": self.efs_grid.value(), "berry_step": self.berry_step.value(), "sampling_mode": self.berry_sampling.currentData() or "single_plaquette", "berry_record_path": self.source_result.currentData(), "frequency_window": _optional_numbers(self.frequency_window.text(), count=2), "frequency_samples": _optional_numbers(self.frequency_samples.text()), "response_weights": _optional_numbers(self.response_weights.text()), "occupation": _optional_numbers(self.occupation.text()), "plot_after": self.plot_after.isChecked()})
        value["composite_bands_one_based"] = bands; value["berry_target_mode"] = target_mode; calc["operation"] = operation

    def _sync_plot(self) -> None:
        def limits(low: QtWidgets.QLineEdit, high: QtWidgets.QLineEdit):
            if not low.text().strip() and not high.text().strip(): return None
            if not low.text().strip() or not high.text().strip(): raise ValueError("Both axis limit values are required")
            values = [safe_number(low.text()), safe_number(high.text())]
            if values[0] >= values[1]: raise ValueError("Axis minimum must be smaller than maximum")
            return values
        vmin = None if not self.vmin.text().strip() else safe_number(self.vmin.text()); vmax = None if not self.vmax.text().strip() else safe_number(self.vmax.text())
        if vmin is not None and vmax is not None and vmin >= vmax: raise ValueError("Color minimum must be smaller than maximum")
        update = {"frequency_unit": self.unit.currentText(), "band_line": self.band_lines.isChecked(), "band_markers": self.band_markers.isChecked(), "grid": self.grid.isChecked(), "legend": self.legend.isChecked(), "berry_render_mode": self.render_mode.currentData(), "cmap": self.cmap.currentText(), "colorbar": self.colorbar.isChecked(), "show_sample_centers": self.sample_centers.isChecked(), "width_px": self.width.value(), "height_px": self.height.value(), "dpi": self.dpi.value(), "title": self.title_edit.text(), "x_label": self.x_label_edit.text(), "y_label": self.y_label_edit.text(), "x_limits": limits(self.xmin, self.xmax), "y_limits": limits(self.ymin, self.ymax), "linewidth": self.linewidth.value(), "marker_size": self.marker_size.value(), "component_index": self.component.value(), "field_quantity": self.field_quantity.currentText(), "berry_vmin": vmin, "berry_vmax": vmax, "interpolation_resolution": self.interpolation_resolution.value(), "efs_levels": _optional_numbers(self.efs_levels.text())}
        self.project["plot"].update(update)
        result = self._selected_result_entry()
        if result is not None: result.setdefault("plot", {}).update(update)

    def _model_changed(self, *_args) -> None:
        if self._building: return
        try:
            self._sync_model(); self.dirty = True; self.statusBar().showMessage("Geometry changed — pending Refresh/Run")
        except ValueError as exc:
            self.statusBar().showMessage(f"Geometry input: {exc}")

    def _calculation_changed(self, *_args) -> None:
        if self._building: return
        try:
            if self.operation.currentData() == "berry":
                target = self.band.value() if self.berry_target.currentData() == "single_band" else self.last_band.value()
                blocker = QtCore.QSignalBlocker(self.numeig); self.numeig.setValue(target + 1); del blocker
            self._sync_calculation(); self._update_calculation_visibility(); self.dirty = True; self.statusBar().showMessage("Calculation changed")
        except ValueError as exc: self.statusBar().showMessage(f"Calculation input: {exc}")

    def _plot_changed(self, *_args) -> None:
        if self._building: return
        try:
            self._sync_plot(); self._update_plot_visibility(); self.dirty = True; self.statusBar().showMessage("Pending plot update")
        except ValueError as exc: self.statusBar().showMessage(f"Plot input: {exc}")

    def edit_sites(self) -> None:
        self._sync_model(); dialog = SitesDialog(self, self.project["model"], self.material_rep.currentText())
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
            style = {**self.project["plot"], **result.get("plot", {})}
            self.current_view = record_view(path, frequency_unit=style.get("frequency_unit", "Normalized"), component_index=int(style.get("component_index", 0)), field_quantity=style.get("field_quantity", "energy_density"))
            pins = result.get("display_state", {}).get("pins", [])
            self.result_canvas.set_record(self.current_view, style, pins=pins); self.inspector.set_view(self.current_view); self._populate_layers(); self._update_plot_visibility(self.current_view.operation); self.result_details.setPlainText(json.dumps({"record": str(path), "operation": self.current_view.operation, "summary": self.current_view.summary, "calculation": result.get("calculation_snapshot"), "model": result.get("model_snapshot")}, indent=2, ensure_ascii=False, default=str)); self.central_tabs.setCurrentWidget(self.result_canvas); self.statusBar().showMessage(f"{self.current_view.operation} · {path}")
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

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Delete:
            rows = self.inspector.selectionModel().selectedRows()
            if rows and rows[0].row() in self.result_canvas.pinned_rows(): self.result_canvas.pin_row(rows[0].row())
            return
        super().keyPressEvent(event)

    def _populate_layers(self) -> None:
        while self.layers_layout.count() > 1:
            item = self.layers_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for name in self.result_canvas.layer_names():
            check = QtWidgets.QCheckBox(name); check.setChecked(True); check.toggled.connect(lambda visible, layer=name: self.result_canvas.set_layer_visible(layer, visible)); self.layers_layout.insertWidget(self.layers_layout.count() - 1, check)

    def add_calculation(self) -> None:
        entry = add_calculation(self.project, name="New calculation", operation="frequency_at_k")
        self.project["selected_node"] = {"kind": "calculation", "id": entry["id"]}; self.dirty = True; self._insert_calculation_item(entry); self._load_calculation_controls()

    def copy_calculation(self) -> None:
        source = self._selected_calculation(); entry = copy_calculation(self.project, source["id"])
        self.project["selected_node"] = {"kind": "calculation", "id": entry["id"]}; self.dirty = True; self._insert_calculation_item(entry); self._load_calculation_controls()

    def rename_calculation(self) -> None:
        entry = self._selected_calculation(); name, ok = QtWidgets.QInputDialog.getText(self, "Rename calculation", "Name", text=entry.get("name", "Calculation"))
        if ok and name.strip():
            rename_calculation(self.project, entry["id"], name.strip()); self.dirty = True
            item = self._tree_item("calculation", entry["id"])
            if item is not None: item.setText(0, name.strip())

    def delete_calculation(self) -> None:
        try:
            identity = self._selected_calculation()["id"]; item = self._tree_item("calculation", identity); delete_calculation(self.project, identity); self.dirty = True
            if item is not None and item.parent() is not None: item.parent().removeChild(item)
            self._load_calculation_controls()
        except ValueError as exc: QtWidgets.QMessageBox.warning(self, "Cannot delete calculation", str(exc))

    def _tree_item(self, kind: str, identity: str) -> QtWidgets.QTreeWidgetItem | None:
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == (kind, identity): return item
            iterator += 1
        return None

    def _insert_calculation_item(self, entry: dict[str, Any]) -> None:
        root = self.tree.topLevelItem(1); item = QtWidgets.QTreeWidgetItem([entry.get("name", OPERATIONS.get(entry["operation"], entry["operation"]))]); item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("calculation", entry["id"])); root.addChild(item); root.setExpanded(True); self.tree.setCurrentItem(item)

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
        style = {**self.project["plot"], **result.get("plot", {})}; style["x_limits"], style["y_limits"] = self.result_canvas.view_limits()
        figure = plot_record(path, style); export_figure(figure, target, width_px=style["width_px"], height_px=style["height_px"], dpi=style["dpi"]); self.statusBar().showMessage(f"Exported {target}")

    def run(self) -> None:
        if self.process is not None: return
        try:
            if self.project_path is None and not self.save_as(): return
            self._sync_model(); self._sync_calculation(); self._sync_plot(); validate_project(self.project); self.save()
            request = build_worker_request(self.project, self.project_path); handle, name = tempfile.mkstemp(prefix="legumephc-qt-", suffix=".json"); import os; os.close(handle); self.request_path = Path(name); self.request_path.write_text(json.dumps(request), encoding="utf-8"); self.active_request = deepcopy(request)
            self.process = QtCore.QProcess(self); self.process.setWorkingDirectory(str(ROOT)); self.process.setProgram(sys.executable); self.process.setArguments(["-m", "legumephc.studio.worker", "--request", str(self.request_path)]); self.process.readyReadStandardOutput.connect(self._read_worker); self.process.readyReadStandardError.connect(self._read_worker_error); self.process.finished.connect(self._worker_finished); self.stdout_buffer = ""; self.run_started = time.monotonic(); self.last_worker_error = None; self._cancelling = False; self.progress.setRange(0, 0); self.progress_label.setText(f"Starting {request['calculation']['operation']}"); self.process.start()
            self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
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
            else: self.progress.setRange(0, 0); self.progress_label.setText(str(event.get("message", event.get("error", event["event"]))))
            if event.get("event") in {"phase", "record_written", "failed"}: self.log.appendPlainText(str(event.get("message", event.get("error", event["event"]))))
            if event.get("event") == "failed": self.last_worker_error = str(event.get("error", "Worker failed"))
            if event.get("event") == "completed": self._accept_worker_result(event)

    def _read_worker_error(self) -> None:
        if self.process is not None: self.log.appendPlainText(bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip())

    def _accept_worker_result(self, event: dict[str, Any]) -> None:
        assert self.project_path is not None and self.active_request is not None
        reference = record_reference(Path(event["record_path"]), self.project_path.parent); calculation_id = self.active_request.get("calculation_id", self._selected_calculation()["id"]); result_id = f"result-{len(self.project.get('results', [])) + 1}"
        self.project["results"].append({"id": result_id, "calculation_id": calculation_id, "record_reference": reference, "model_snapshot": self.active_request["model_snapshot"], "calculation_snapshot": self.active_request["calculation_snapshot"], "plot": deepcopy(self.project["plot"]), "display_state": {"pins": []}}); self.project["selected_result"] = reference["path"]; self.dirty = True; self._populate_tree(); self._populate_berry_sources()
        if self.active_request["calculation_snapshot"].get("plot_after", True): self.plot_selected()

    def _worker_finished(self, exit_code: int, _status) -> None:
        if self._cancelling: self.progress_label.setText("Cancelled; exact worker terminated")
        elif exit_code != 0: self.progress_label.setText(self.last_worker_error or f"Worker failed with exit code {exit_code}")
        else: self.progress.setRange(0, 100); self.progress.setValue(100); self.progress_label.setText(f"Completed in {time.monotonic() - self.run_started:.1f} s")
        self._cleanup_worker()

    def cancel(self) -> None:
        if self.process is None: return
        self._cancelling = True
        self.process.terminate()
        if not self.process.waitForFinished(5000): self.process.kill(); self.process.waitForFinished(5000)
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress_label.setText("Cancelled; exact worker terminated"); self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self.request_path is not None: self.request_path.unlink(missing_ok=True)
        self.request_path = None; self.process = None; self.active_request = None
        if hasattr(self, "run_button"): self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)

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
    app.setEffectEnabled(QtCore.Qt.UIEffect.UI_AnimateCombo, False)
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    window = StudioWindow(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
