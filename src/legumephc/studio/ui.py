from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np

from .plotting import default_plot_style, export_figure, plot_record
from .preview import preview_geometry
from .profile import model_from_case
from .geometry_editor import (
    SHAPE_CHOICES,
    canonical_shape,
    editor_value,
    epsilon_from_editor,
    model_name,
    safe_number,
    shape_choice,
    uniaxial_matrix,
)
from .project import (
    PROJECT_SUFFIX,
    apply_preset,
    load_preset,
    load_project,
    new_project,
    new_preset,
    record_available,
    record_reference,
    project_records_dir,
    save_project,
    save_preset,
    validate_project,
)
from .worker import WorkerProcess, build_worker_request, start_worker


ROOT = Path(__file__).resolve().parents[3]
PROJECTS_DIR = ROOT / "projects"
PRESETS_DIR = ROOT / "presets"
FIGURES_DIR = ROOT / "figures"


def _control_text(value: Any) -> Any:
    """Normalize nullable plot values before assigning them to Tk variables."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    return value


def _dialog_location(app: "StudioApp", kind: str, *, saving: bool = False, record_path: str | None = None) -> tuple[str, str]:
    """Return an explicit initial directory and filename for every Studio dialog."""
    project_parent = app.project_path.parent if app.project_path else None
    if kind == "project":
        directory = project_parent or PROJECTS_DIR
        filename = app.project_path.name if app.project_path else f"Untitled{PROJECT_SUFFIX}"
    elif kind == "preset":
        directory = (project_parent / "presets") if project_parent else PRESETS_DIR
        filename = "Untitled.legumephc-preset.json"
    else:
        record_dir = None
        if record_path and project_parent:
            reference_path = None
            for result in app.project.get("results", []):
                reference = result.get("record_reference", {})
                if result.get("id") == record_path or reference.get("path") == record_path:
                    reference_path = reference.get("path")
                    break
            if reference_path:
                record_dir = project_parent / reference_path / "figures"
        directory = record_dir or ((project_parent / "figures") if project_parent else FIGURES_DIR)
        filename = "result.png"
    if saving:
        directory.mkdir(parents=True, exist_ok=True)
    return str(directory), filename


class MotifDialog(tk.Toplevel):
    """One structured editor for a motif; no geometry syntax is typed by hand."""

    def __init__(self, parent: tk.Misc, *, motif: dict[str, Any], representation: str, title: str):
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.resizable(False, False)
        self.result: dict[str, Any] | None = None
        self.representation = representation
        center = motif.get("center", [0.0, 0.0])
        self.variables = {
            "name": tk.StringVar(self, str(motif.get("name", ""))),
            "shape": tk.StringVar(self, shape_choice(str(motif.get("kind", "circle")), motif.get("sides"))),
            "radius": tk.StringVar(self, str(motif.get("radius", 0.2))),
            "center_x": tk.StringVar(self, str(center[0])),
            "center_y": tk.StringVar(self, str(center[1])),
            "material": tk.StringVar(self, str(editor_value(float(motif.get("epsilon", 1.0)), representation))),
            "sides": tk.StringVar(self, str(motif.get("sides", 6))),
            "angle": tk.StringVar(self, str(motif.get("angle_degrees", 0.0))),
        }
        body = ttk.Frame(self, padding=10)
        body.grid(sticky="nsew")
        ttk.Label(body, text="Shape").grid(row=0, column=0, sticky="w", pady=3)
        self.shape_menu = ttk.Combobox(body, textvariable=self.variables["shape"], values=SHAPE_CHOICES, state="readonly", width=20)
        self.shape_menu.grid(row=0, column=1, sticky="ew", pady=3)
        self.shape_menu.bind("<<ComboboxSelected>>", lambda _event: self._shape_changed())
        labels = (
            ("Name (optional)", "name"),
            ("Radius / circumradius r/a", "radius"),
            ("Center x/a", "center_x"),
            ("Center y/a", "center_y"),
            ("Refractive index n" if representation == "n" else "Epsilon ε", "material"),
            ("Polygon sides", "sides"),
            ("Rotation (deg)", "angle"),
        )
        self.entries: dict[str, ttk.Entry] = {}
        for row, (label, name) in enumerate(labels, start=1):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(body, textvariable=self.variables[name], width=22)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.entries[name] = entry
        self.error = ttk.Label(body, text="", foreground="#b00020", wraplength=330)
        self.error.grid(row=8, column=0, columnspan=2, sticky="w", pady=(5, 2))
        buttons = ttk.Frame(body)
        buttons.grid(row=9, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply", command=self._apply).pack(side="right", padx=(0, 6))
        self.bind("<Return>", lambda _event: self._apply())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._shape_changed()
        self.grab_set()
        self.wait_visibility()
        self.focus_set()

    def _shape_changed(self) -> None:
        choice = self.variables["shape"].get()
        regular = choice == "Regular polygon"
        circle = choice == "Circle"
        self.entries["sides"].configure(state="normal" if regular else "disabled")
        self.entries["angle"].configure(state="disabled" if circle else "normal")
        if choice == "Triangle":
            self.variables["sides"].set("3")
        elif choice == "Square":
            self.variables["sides"].set("4")

    def _apply(self) -> None:
        try:
            kind, sides = canonical_shape(self.variables["shape"].get(), self.variables["sides"].get())
            radius = safe_number(self.variables["radius"].get())
            if radius <= 0:
                raise ValueError("radius must be positive")
            center = [safe_number(self.variables["center_x"].get()), safe_number(self.variables["center_y"].get())]
            epsilon = epsilon_from_editor(self.variables["material"].get(), self.representation)
            angle = 0.0 if kind == "circle" else safe_number(self.variables["angle"].get())
            default_name = self.variables["shape"].get().replace(" ", "")
            self.result = {
                "name": self.variables["name"].get().strip() or default_name,
                "kind": kind,
                "radius": radius,
                "center": center,
                "epsilon": epsilon,
                "sides": sides,
                "angle_degrees": angle,
            }
        except (ValueError, TypeError) as exc:
            self.error.configure(text=str(exc))
            return
        self.destroy()


class StudioApp(tk.Tk):
    """Small single-window Studio; all formal solves run in one child."""

    def __init__(self, project: dict[str, Any] | None = None, project_path: str | Path | None = None):
        super().__init__()
        self.project = project or new_project()
        self.project_path = Path(project_path).resolve() if project_path else None
        self.dirty = False
        self.worker: WorkerProcess | None = None
        self._active_request: dict[str, Any] | None = None
        self._run_started_at: float | None = None
        self.figure = None
        self.canvas = None
        self.preview_figure = None
        self.preview_canvas = None
        self.toolbar = None
        self.preview_toolbar = None
        self.plot_host = None
        self._vars: dict[str, tk.Variable] = {}
        self._field_widgets: dict[str, list[tk.Widget]] = {}
        self._calculation_field_names = {"qx", "qy", "band", "composite", "gmax", "numeig", "field_grid", "efs_grid", "berry_step", "samples", "berry_grid", "source_result", "response_weights"}
        self._style_widgets: dict[str, list[tk.Widget]] = {}
        self._build()
        for name in ("width_px", "height_px", "dpi", "title", "x_label", "y_label", "x_limits", "y_limits", "linewidth", "marker_size", "cmap", "component_index", "field_quantity", "grid", "legend", "band_line", "band_markers", "berry_coloring", "berry_interpolation", "colorbar", "berry_vmin", "berry_vmax"):
            if name in self._vars:
                self._vars[name].trace_add("write", self._mark_style_pending)
        self._populate()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        self._update_title()
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New Project", command=self._new)
        file_menu.add_command(label="Open Project…", command=self._open)
        file_menu.add_command(label="Save", command=self._save)
        file_menu.add_command(label="Save As…", command=self._save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Load Parameter Preset…", command=self._load_preset)
        file_menu.add_command(label="Save Parameter Preset…", command=self._save_preset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._close)
        menu.add_cascade(label="Project", menu=file_menu)
        self.config(menu=menu)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)
        navigator = ttk.Frame(main, width=190)
        self.tree = ttk.Treeview(navigator, show="tree", selectmode="browse")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        main.add(navigator, weight=0)
        notebook = ttk.Notebook(main)
        self.geometry_tab = ttk.Frame(notebook)
        self.calculation_tab = ttk.Frame(notebook)
        self.results_tab = ttk.Frame(notebook)
        self.style_tab = ttk.Frame(notebook)
        notebook.add(self.geometry_tab, text="Geometry")
        notebook.add(self.calculation_tab, text="Calculation")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.style_tab, text="Plot Style")
        main.add(notebook, weight=3)
        self.side_notebook = ttk.Notebook(main)
        self.geometry_side = ttk.Frame(self.side_notebook)
        self.result_side = ttk.Frame(self.side_notebook)
        self.side_notebook.add(self.geometry_side, text="Geometry")
        self.side_notebook.add(self.result_side, text="Result")
        self.side_status = ttk.Label(self.geometry_side, text="Solver-free geometry preview")
        self.side_status.pack(anchor="nw", padx=6, pady=6)
        self.side_result_status = ttk.Label(self.result_side, text="Select a result to plot")
        self.side_result_status.pack(anchor="nw", padx=6, pady=6)
        main.add(self.side_notebook, weight=2)
        self._build_geometry_tab()
        self._build_calculation_tab()
        self._build_results_tab()
        self._build_style_tab()
        self.log = tk.Text(self, height=5, state="disabled")
        self.log.pack(fill="x", padx=6, pady=(0, 6))
        self._loading = False
        for name in ("lattice", "lattice_constant", "material_representation", "background_material", "deformation_kind", "deformation_factor", "deformation_angle", "b11", "b12", "b21", "b22", "a11", "a12", "a21", "a22", "tx", "ty"):
            self._vars[name].trace_add("write", self._mark_pending)

    def _var(self, name: str, value: Any = "") -> tk.Variable:
        variable = tk.StringVar(self, str(value))
        self._vars[name] = variable
        return variable

    def _bool_var(self, name: str, value: bool = False) -> tk.BooleanVar:
        variable = tk.BooleanVar(self, value=value)
        self._vars[name] = variable
        return variable

    def _label_entry(self, parent, row: int, label: str, name: str, width: int = 12) -> None:
        label_widget = ttk.Label(parent, text=label)
        entry_widget = ttk.Entry(parent, textvariable=self._var(name), width=width)
        label_widget.grid(row=row, column=0, sticky="w", padx=4, pady=2)
        entry_widget.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        self._field_widgets[name] = [label_widget, entry_widget]

    def _build_geometry_tab(self) -> None:
        controls = ttk.LabelFrame(self.geometry_tab, text="Geometry editor")
        controls.pack(side="left", fill="y", padx=6, pady=6)
        self._geometry_controls = controls
        ttk.Label(controls, text="Lattice").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.lattice_menu = ttk.Combobox(controls, textvariable=self._var("lattice"), values=("triangular", "square", "custom"), state="readonly", width=13)
        self.lattice_menu.grid(row=0, column=1, padx=4, pady=2)
        self.lattice_menu.bind("<<ComboboxSelected>>", lambda _event: self._lattice_changed())
        self._label_entry(controls, 1, "Lattice constant a", "lattice_constant")
        ttk.Label(controls, text="Direct basis").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.basis_summary = ttk.Label(controls, text="automatic for lattice", wraplength=190)
        self.basis_summary.grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(controls, text="Material representation").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.material_menu = ttk.Combobox(controls, textvariable=self._var("material_representation"), values=("epsilon", "n"), state="readonly", width=13)
        self.material_menu.grid(row=3, column=1, sticky="w", padx=4, pady=2)
        self.material_menu.bind("<<ComboboxSelected>>", lambda _event: self._material_changed())
        self._label_entry(controls, 4, "Background", "background_material")
        ttk.Label(controls, text="Motifs").grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 2))
        self.motif_tree = ttk.Treeview(controls, columns=("name", "kind", "radius", "center", "material", "sides", "angle"), show="headings", height=5)
        for column, heading, width in (("name", "Name", 82), ("kind", "Kind", 58), ("radius", "R/a", 42), ("center", "Center", 92), ("material", "E/n", 55), ("sides", "Sides", 42), ("angle", "Angle", 48)):
            self.motif_tree.heading(column, text=heading)
            self.motif_tree.column(column, width=width, stretch=False)
        self.motif_tree.grid(row=6, column=0, columnspan=2, padx=4, pady=2)
        self.motif_tree.bind("<<TreeviewSelect>>", lambda _event: self._motif_selected())
        self.motif_tree.bind("<Double-1>", lambda _event: self._edit_motif())
        motif_buttons = ttk.Frame(controls)
        motif_buttons.grid(row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Button(motif_buttons, text="Add", command=self._add_motif).pack(side="left")
        ttk.Button(motif_buttons, text="Edit", command=self._edit_motif).pack(side="left", padx=2)
        ttk.Button(motif_buttons, text="Remove", command=self._remove_motif).pack(side="left")
        ttk.Label(controls, text="Deformation").grid(row=8, column=0, sticky="w", padx=4, pady=(8, 2))
        self.deformation_menu = ttk.Combobox(controls, textvariable=self._var("deformation_kind"), values=("none", "uniaxial", "custom"), state="readonly", width=13)
        self.deformation_menu.grid(row=8, column=1, sticky="w", padx=4, pady=2)
        self.deformation_menu.bind("<<ComboboxSelected>>", lambda _event: self._deformation_changed())
        self._label_entry(controls, 9, "Uniaxial factor", "deformation_factor")
        self._label_entry(controls, 10, "Uniaxial angle", "deformation_angle")
        self.advanced_button = ttk.Button(controls, text="Advanced ▸", command=self._toggle_advanced)
        self.advanced_button.grid(row=11, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 2))
        self.advanced_frame = ttk.LabelFrame(controls, text="Advanced values")
        self._basis_entries = []
        self._label_entry(self.advanced_frame, 0, "basis 11", "b11")
        self._label_entry(self.advanced_frame, 1, "basis 12", "b12")
        self._label_entry(self.advanced_frame, 2, "basis 21", "b21")
        self._label_entry(self.advanced_frame, 3, "basis 22", "b22")
        self._basis_entries = [self._field_widgets[name][1] for name in ("b11", "b12", "b21", "b22")]
        for row, name in enumerate(("a11", "a12", "a21", "a22"), start=4):
            self._label_entry(self.advanced_frame, row, name, name)
        self._label_entry(self.advanced_frame, 8, "translation dx/a", "tx")
        self._label_entry(self.advanced_frame, 9, "translation dy/a", "ty")
        ttk.Label(self.advanced_frame, text="translation moves motifs only; lattice unchanged", wraplength=190).grid(row=10, column=0, columnspan=2, padx=4, pady=2)
        self.preview_status = ttk.Label(controls, text="Pending Refresh")
        self.preview_status.grid(row=12, column=0, columnspan=2, sticky="w", padx=4)
        self.geometry_refresh_button = ttk.Button(controls, text="Refresh Geometry", command=self._refresh_preview)
        self.geometry_refresh_button.grid(row=13, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        self._build_geometry_views()

    def _lattice_changed(self) -> None:
        lattice = self._vars["lattice"].get()
        if lattice == "triangular":
            summary = "a1=(a/2, √3a/2), a2=(a/2, −√3a/2)"
            basis = ((0.5, 0.5), (3.0 ** 0.5 / 2.0, -3.0 ** 0.5 / 2.0))
        elif lattice == "square":
            summary = "a1=(a, 0), a2=(0, a)"
            basis = ((1.0, 0.0), (0.0, 1.0))
        else:
            summary = "custom basis (dimensionless)"
            basis = None
        self.basis_summary.configure(text=summary)
        for entry in self._basis_entries:
            entry.configure(state="disabled" if basis is not None else "normal")
        if basis is not None:
            for name, value in zip(("b11", "b12", "b21", "b22"), (basis[0][0], basis[0][1], basis[1][0], basis[1][1])):
                self._vars[name].set(value)

    def _deformation_changed(self) -> None:
        kind = self._vars["deformation_kind"].get()
        state = "normal" if kind == "uniaxial" else "disabled"
        for name in ("deformation_factor", "deformation_angle"):
            self._field_widgets[name][1].configure(state=state)
        custom_state = "normal" if kind == "custom" else "disabled"
        for name in ("a11", "a12", "a21", "a22"):
            self._field_widgets[name][1].configure(state=custom_state)

    def _toggle_advanced(self) -> None:
        expanded = not bool(self.project.get("ui_state", {}).get("advanced_expanded", False))
        self._set_advanced(expanded)

    def _set_advanced(self, expanded: bool) -> None:
        self.project.setdefault("ui_state", {})["advanced_expanded"] = expanded
        self.advanced_button.configure(text="Advanced ▾" if expanded else "Advanced ▸")
        if expanded:
            self.advanced_frame.grid(row=12, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
            self.preview_status.grid(row=13, column=0, columnspan=2, sticky="w", padx=4)
            self.geometry_refresh_button.grid(row=14, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        else:
            self.advanced_frame.grid_remove()
            self.preview_status.grid(row=12, column=0, columnspan=2, sticky="w", padx=4)
            self.geometry_refresh_button.grid(row=13, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        self.dirty = True

    def _geometry_tab_changed(self) -> None:
        if not hasattr(self, "geometry_notebook"):
            return
        tab_id = self.geometry_notebook.select()
        self.project.setdefault("ui_state", {})["active_geometry_tab"] = self.geometry_notebook.tab(tab_id, "text")

    def _motif_selected(self) -> None:
        selection = self.motif_tree.selection()
        if selection:
            self.project.setdefault("ui_state", {})["selected_motif"] = selection[0]

    def _material_changed(self) -> None:
        representation = self._vars["material_representation"].get()
        self.project.setdefault("ui_state", {})["material_representation"] = representation
        self._loading = True
        self._populate_motifs()
        geometry = self.project["case"]["geometry"]
        self._vars["background_material"].set(editor_value(geometry.get("epsilon_background", 7.29), representation))
        self._loading = False
        self.dirty = True
        self.preview_status.configure(text="Pending Refresh")

    def _build_geometry_views(self) -> None:
        if getattr(self, "side_status", None) is not None:
            self.side_status.destroy()
        self.geometry_notebook = ttk.Notebook(self.geometry_side)
        self.geometry_notebook.pack(fill="both", expand=True)
        self.geometry_hosts = {}
        for name in ("Motif", "Unit Cell", "Motif Array", "Lattice Sites", "Reciprocal & BZ", "Epsilon"):
            host = ttk.Frame(self.geometry_notebook)
            self.geometry_notebook.add(host, text=name)
            self.geometry_hosts[name] = host
        self.geometry_notebook.bind("<<NotebookTabChanged>>", lambda _event: self._geometry_tab_changed())

    def _build_calculation_tab(self) -> None:
        controls = ttk.LabelFrame(self.calculation_tab, text="Calculation settings")
        controls.pack(anchor="nw", padx=6, pady=6)
        ttk.Label(controls, text="Calculation").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.operation_menu = ttk.Combobox(controls, textvariable=self._var("operation"), values=("frequency_at_k", "band_structure", "fields_energy", "efs", "berry", "berry_curvature_dipole"), state="readonly", width=24)
        self.operation_menu.grid(row=0, column=1, padx=4, pady=2)
        self.operation_menu.bind("<<ComboboxSelected>>", lambda _event: self._operation_changed())
        for row, label, name in ((1, "q x", "qx"), (2, "q y", "qy"), (3, "Band (one-based)", "band"), (4, "Composite bands (one-based)", "composite"), (5, "gmax", "gmax"), (6, "numeig", "numeig"), (7, "Field grid size", "field_grid"), (8, "EFS grid size", "efs_grid"), (9, "Berry step", "berry_step")):
            self._label_entry(controls, row, label, name, 24 if name == "composite" else 12)
        ttk.Label(controls, text="Polarization").grid(row=10, column=0, sticky="w", padx=4, pady=2)
        self.polarization_widget = ttk.Combobox(controls, textvariable=self._var("polarization"), values=("te", "tm"), state="readonly", width=12)
        self.polarization_widget.grid(row=10, column=1, sticky="w", padx=4, pady=2)
        self._label_entry(controls, 11, "Path samples/segment", "samples", 12)
        self.berry_sampling_label = ttk.Label(controls, text="Berry sampling")
        self.berry_sampling_label.grid(row=12, column=0, sticky="w", padx=4, pady=2)
        self.berry_sampling_widget = ttk.Combobox(controls, textvariable=self._var("berry_sampling"), values=("single_plaquette", "first_bz_grid", "explicit_centers"), state="readonly", width=20)
        self.berry_sampling_widget.grid(row=12, column=1, sticky="w", padx=4, pady=2)
        self._label_entry(controls, 13, "Berry grid size", "berry_grid", 12)
        buttons = ttk.Frame(controls)
        buttons.grid(row=14, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        self.run_button = ttk.Button(buttons, text="Run", command=self._run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=4)
        ttk.Button(buttons, text="Add", command=self._add_calculation).pack(side="left", padx=2)
        ttk.Button(buttons, text="Copy", command=self._copy_calculation).pack(side="left", padx=2)
        ttk.Button(buttons, text="Rename", command=self._rename_calculation).pack(side="left", padx=2)
        ttk.Button(buttons, text="Delete", command=self._delete_calculation).pack(side="left", padx=2)
        self.plot_after = ttk.Checkbutton(buttons, text="Plot after Run", variable=self._plot_after_var)
        self.plot_after.pack(side="left")
        self.progress_bar = ttk.Progressbar(controls, mode="determinate", maximum=100)
        self.progress_bar.grid(row=15, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 0))
        self.calc_status = ttk.Label(controls, text="Ready", wraplength=410)
        self.calc_status.grid(row=16, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 0))
        self.calc_detail = ttk.Label(controls, text="", wraplength=410, foreground="#555555")
        self.calc_detail.grid(row=17, column=0, columnspan=2, sticky="w", padx=4)
        self._label_entry(controls, 18, "Berry source result", "source_result", 28)
        self._label_entry(controls, 19, "Response weights", "response_weights", 28)
        self._operation_changed()

    def _operation_changed(self) -> None:
        operation_var = self._vars.get("operation")
        operation = operation_var.get() if operation_var is not None else "frequency_at_k"
        visible = {"gmax", "numeig", "polarization"}
        visible.update({
            "frequency_at_k": {"qx", "qy", "band"},
            "band_structure": {"samples"},
            "fields_energy": {"qx", "qy", "composite", "field_grid"},
            "efs": {"composite", "efs_grid"},
            "berry": {"composite", "berry_step", "berry_grid", "berry_sampling"},
            "berry_curvature_dipole": set(),
        }.get(operation, set()))
        if operation == "berry_curvature_dipole":
            visible.update({"source_result", "response_weights"})
        for name, widgets in self._field_widgets.items():
            if name not in self._calculation_field_names:
                continue
            for widget in widgets:
                if name in visible:
                    widget.grid()
                else:
                    widget.grid_remove()
        self.polarization_widget.grid() if "polarization" in visible else self.polarization_widget.grid_remove()
        if hasattr(self, "berry_sampling_widget"):
            (self.berry_sampling_widget.grid if "berry_sampling" in visible else self.berry_sampling_widget.grid_remove)()
            (self.berry_sampling_label.grid if "berry_sampling" in visible else self.berry_sampling_label.grid_remove)()
        self._update_style_visibility(operation)

    def _build_results_tab(self) -> None:
        frame = ttk.Frame(self.results_tab)
        frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.result_list = tk.Listbox(frame, height=12)
        self.result_list.pack(side="left", fill="both", expand=True)
        self.result_list.bind("<<ListboxSelect>>", lambda _event: self._select_result())
        buttons = ttk.Frame(frame)
        buttons.pack(side="left", fill="y", padx=6)
        ttk.Button(buttons, text="Refresh", command=self._populate_results).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Plot selected", command=self._plot_selected).pack(fill="x", pady=2)
        self.result_status = ttk.Label(buttons, text="No result selected", wraplength=220)
        self.result_status.pack(anchor="w", pady=8)

    def _build_style_tab(self) -> None:
        controls = ttk.LabelFrame(self.style_tab, text="Figure style")
        controls.pack(anchor="nw", padx=6, pady=6)
        for row, label, name in ((0, "Width px", "width_px"), (1, "Height px", "height_px"), (2, "DPI", "dpi"), (3, "Title", "title"), (4, "X label", "x_label"), (5, "Y label", "y_label")):
            self._label_entry(controls, row, label, name, 28 if name in {"title", "x_label", "y_label"} else 12)
        for row, label, name in ((6, "X limits", "x_limits"), (7, "Y limits", "y_limits"), (8, "Line width", "linewidth"), (9, "Marker size", "marker_size"), (10, "Color map", "cmap"), (11, "Component / band", "component_index")):
            self._label_entry(controls, row, label, name, 28 if name in {"x_limits", "y_limits", "cmap"} else 12)
        self._style_bool(controls, 12, "Band lines", "band_line")
        self._style_bool(controls, 13, "Band markers", "band_markers")
        ttk.Label(controls, text="Field quantity").grid(row=14, column=0, sticky="w", padx=4, pady=2)
        field_quantity = ttk.Combobox(controls, textvariable=self._var("field_quantity"), values=("E2", "H2", "energy_density"), state="readonly", width=16)
        field_quantity.grid(row=14, column=1, sticky="w", padx=4, pady=2)
        self._style_widgets["field_quantity"] = [controls.grid_slaves(row=14, column=0)[0], field_quantity]
        self._style_bool(controls, 15, "Grid", "grid")
        self._style_bool(controls, 16, "Legend", "legend")
        self._style_bool(controls, 17, "Berry samples", "berry_coloring")
        self._style_bool(controls, 18, "Berry interpolation", "berry_interpolation")
        self._style_bool(controls, 19, "Colorbar", "colorbar")
        self._label_entry(controls, 20, "Berry vmin", "berry_vmin", 12)
        self._label_entry(controls, 21, "Berry vmax", "berry_vmax", 12)
        self._style_widgets["berry_vmin"] = self._field_widgets.pop("berry_vmin")
        self._style_widgets["berry_vmax"] = self._field_widgets.pop("berry_vmax")
        for name in ("width_px", "height_px", "dpi", "title", "x_label", "y_label", "x_limits", "y_limits", "linewidth", "marker_size", "cmap", "component_index"):
            self._style_widgets[name] = self._field_widgets.pop(name)
        ttk.Button(controls, text="Apply to Current Plot", command=self._apply_plot_style).grid(row=22, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 2))
        self.style_status = ttk.Label(controls, text="Style current", foreground="#555555")
        self.style_status.grid(row=23, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Button(controls, text="Export current plot…", command=self._export).grid(row=24, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        operation_var = self._vars.get("operation")
        self._update_style_visibility(operation_var.get() if operation_var is not None else "frequency_at_k")

    def _style_bool(self, parent, row: int, label: str, name: str) -> None:
        variable = self._bool_var(name)
        widget = ttk.Checkbutton(parent, text=label, variable=variable)
        widget.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        self._style_widgets[name] = [widget]

    def _update_style_visibility(self, operation: str | None = None) -> None:
        operation = operation or "frequency_at_k"
        public = {"width_px", "height_px", "dpi", "title", "x_label", "y_label", "grid", "legend", "x_limits", "y_limits"}
        specific = {
            "solve_bands": {"band_line", "band_markers", "linewidth", "marker_size"}, "band_structure": {"band_line", "band_markers", "linewidth", "marker_size"},
            "solve_berry": {"cmap", "berry_interpolation", "berry_vmin", "berry_vmax", "colorbar", "berry_coloring"}, "berry": {"cmap", "berry_interpolation", "berry_vmin", "berry_vmax", "colorbar", "berry_coloring"},
            "solve_efs": {"component_index", "cmap", "colorbar"}, "efs": {"component_index", "cmap", "colorbar"},
            "compute_field_observables": {"field_quantity", "component_index", "cmap", "colorbar"}, "fields_energy": {"field_quantity", "component_index", "cmap", "colorbar"},
            "compute_berry_dipole": {"component_index", "cmap", "colorbar"}, "berry_curvature_dipole": {"component_index", "cmap", "colorbar"},
        }.get(operation, set())
        visible = public | specific
        for name, widgets in self._style_widgets.items():
            for widget in widgets:
                (widget.grid if name in visible else widget.grid_remove)()

    @property
    def _plot_after_var(self) -> tk.BooleanVar:
        if "plot_after" not in self._vars:
            self._vars["plot_after"] = tk.BooleanVar(self, True)
        return self._vars["plot_after"]

    def _populate(self) -> None:
        self._loading = True
        case = self.project["case"]
        geometry = case["geometry"]
        basis = case.get("direct_basis", [[1.0, 0.0], [0.0, 1.0]])
        deformation = case.get("deformation", {})
        linear = deformation.get("linear", case.get("affine", {}).get("linear", [[1.0, 0.0], [0.0, 1.0]]))
        translation = deformation.get("translation", case.get("affine", {}).get("translation", [0.0, 0.0]))
        ui_state = self.project.setdefault("ui_state", {})
        values = {
            "lattice": case["lattice"], "lattice_constant": case.get("lattice_constant", 1.0),
            "material_representation": ui_state.get("material_representation", "epsilon"),
            "background_material": editor_value(geometry.get("epsilon_background", 7.29), ui_state.get("material_representation", "epsilon")),
            "deformation_kind": deformation.get("kind", "none"), "deformation_factor": deformation.get("factor", 1.0),
            "deformation_angle": deformation.get("angle_degrees", 0.0), "b11": basis[0][0], "b12": basis[0][1],
            "b21": basis[1][0], "b22": basis[1][1], "a11": linear[0][0], "a12": linear[0][1],
            "a21": linear[1][0], "a22": linear[1][1], "tx": translation[0], "ty": translation[1],
        }
        selected_id = self.project.get("selected_node", {}).get("id")
        calculation = next((item["parameters"] for item in self.project.get("calculations", []) if item["id"] == selected_id), self.project["calculation"])
        values.update({"operation": calculation["operation"], "qx": calculation["qpoint"][0], "qy": calculation["qpoint"][1], "band": calculation["band_one_based"], "composite": ",".join(map(str, calculation["composite_bands_one_based"])), "gmax": calculation["gmax"], "numeig": calculation["numeig"], "polarization": calculation["polarization"], "field_grid": calculation["grid_size"], "efs_grid": calculation["efs_grid_size"], "berry_step": calculation["berry_step"], "samples": calculation.get("samples_per_segment", 16), "berry_sampling": calculation.get("sampling_mode", "single_plaquette"), "berry_grid": calculation.get("grid_size", 3), "source_result": calculation.get("berry_record_path", ""), "response_weights": ",".join(map(str, calculation.get("response_weights") or []))})
        values.update(self.project["plot"])
        for name in ("x_limits", "y_limits"):
            values[name] = _control_text(values.get(name))
        for name in ("berry_vmin", "berry_vmax"):
            if values.get(name) is None:
                values[name] = ""
        for name, value in values.items():
            if name in self._vars:
                self._vars[name].set(bool(value) if isinstance(self._vars[name], tk.BooleanVar) else value)
        self._populate_motifs()
        self._lattice_changed()
        self._deformation_changed()
        if ui_state.get("advanced_expanded", False):
            self._set_advanced(True)
            self.dirty = False
        else:
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text="Advanced ▸")
        if ui_state.get("active_geometry_tab") in self.geometry_hosts:
            self.geometry_notebook.select(list(self.geometry_hosts).index(ui_state["active_geometry_tab"]))
        self._refresh_preview()
        self._populate_results()
        self._populate_tree()
        self._loading = False

    def _populate_tree(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        model_node = self.tree.insert("", "end", iid="model", text="Model / Geometry", open=True)
        self.tree.insert(model_node, "end", iid="geometry", text=self.project.get("model", self.project.get("case", {})).get("geometry", {}).get("name", "Geometry"))
        calcs_node = self.tree.insert("", "end", iid="calculations", text="Calculations", open=True)
        for calculation in self.project.get("calculations", []):
            self.tree.insert(calcs_node, "end", iid=calculation["id"], text=f"{calculation.get('name', calculation['id'])} · {calculation['operation']}")
        results_node = self.tree.insert("", "end", iid="results", text="Results", open=True)
        for result in self.project.get("results", []):
            self.tree.insert(results_node, "end", iid=result.get("id", result.get("record_reference", {}).get("path", "result")), text=result.get("record_reference", {}).get("path", "Result"))
        selected = self.project.get("selected_node", {}).get("id")
        if selected and self.tree.exists(selected):
            self.tree.selection_set(selected)
            self.tree.see(selected)

    def _tree_selected(self, _event=None) -> None:
        selected = self.tree.selection() if hasattr(self, "tree") else ()
        if not selected:
            return
        item = selected[0]
        if item.startswith("calc-"):
            self.project["selected_node"] = {"kind": "calculation", "id": item}
            self._load_selected_controls()
        elif item.startswith("result-"):
            self.project["selected_node"] = {"kind": "result", "id": item}
            result = next((value for value in self.project.get("results", []) if value.get("id") == item), None)
            if result:
                self.project["selected_result"] = result["record_reference"].get("path")
                self.side_result_status.configure(text=f"{result['calculation_id']} · {result['record_reference'].get('path', 'unavailable')}")
                self._update_style_visibility(self._record_operation(result["record_reference"]))
                self.side_notebook.select(self.result_side)
                self._plot_selected()

    def _load_selected_controls(self) -> None:
        entry = self._selected_calculation_entry()
        calculation = entry["parameters"]
        self._loading = True
        values = {"operation": calculation.get("operation", entry.get("operation", "frequency_at_k")), "qx": calculation.get("qpoint", [0.2, 0.07])[0], "qy": calculation.get("qpoint", [0.2, 0.07])[1], "band": calculation.get("band_one_based", 2), "composite": ",".join(map(str, calculation.get("composite_bands_one_based", [2, 3]))), "gmax": calculation.get("gmax", 2), "numeig": calculation.get("numeig", 4), "polarization": calculation.get("polarization", "te"), "field_grid": calculation.get("grid_size", 8), "efs_grid": calculation.get("efs_grid_size", 5), "berry_step": calculation.get("berry_step", 0.02), "samples": calculation.get("samples_per_segment", 16), "berry_sampling": calculation.get("sampling_mode", "single_plaquette"), "berry_grid": calculation.get("grid_size", 3)}
        for name, value in values.items():
            if name in self._vars:
                self._vars[name].set(value)
        self._operation_changed()
        self._loading = False

    def _selected_calculation_entry(self) -> dict[str, Any]:
        selected_id = self.project.get("selected_node", {}).get("id")
        return next((item for item in self.project.get("calculations", []) if item["id"] == selected_id), self.project["calculations"][0])

    def _add_calculation(self) -> None:
        from .project import add_calculation
        add_calculation(self.project)
        self.project["selected_node"] = {"kind": "calculation", "id": self.project["calculations"][-1]["id"]}
        self.dirty = True
        self._populate()

    def _copy_calculation(self) -> None:
        from .project import copy_calculation
        copy_calculation(self.project, self._selected_calculation_entry()["id"])
        self.project["selected_node"] = {"kind": "calculation", "id": self.project["calculations"][-1]["id"]}
        self.dirty = True
        self._populate()

    def _rename_calculation(self) -> None:
        from tkinter import simpledialog
        entry = self._selected_calculation_entry()
        name = simpledialog.askstring("Rename calculation", "Name", initialvalue=entry.get("name", entry["id"]), parent=self)
        if name:
            from .project import rename_calculation
            rename_calculation(self.project, entry["id"], name)
            self.dirty = True
            self._populate_tree()

    def _delete_calculation(self) -> None:
        from .project import delete_calculation
        try:
            delete_calculation(self.project, self._selected_calculation_entry()["id"])
        except ValueError as exc:
            messagebox.showerror("Delete calculation", str(exc))
            return
        self.dirty = True
        self._populate()

    def _mark_pending(self, *_args) -> None:
        if getattr(self, "_loading", True):
            return
        self.dirty = True
        self.preview_status.configure(text="Pending Refresh/Run")
        self.calc_status.configure(text="Pending Run")
        self._update_title()

    def _mark_style_pending(self, *_args) -> None:
        if getattr(self, "_loading", True):
            return
        self.dirty = True
        if hasattr(self, "style_status"):
            self.style_status.configure(text="Pending plot update")
        self._update_title()

    def _case_from_controls(self) -> dict[str, Any]:
        case = dict(self.project["case"])
        def value(name: str, fallback: Any) -> Any:
            variable = self._vars.get(name)
            return variable.get() if variable is not None else fallback

        case["lattice"] = self._vars["lattice"].get()
        case["lattice_constant"] = safe_number(self._vars["lattice_constant"].get())
        if case["lattice"] == "custom":
            case["direct_basis"] = [[safe_number(self._vars["b11"].get()), safe_number(self._vars["b12"].get())], [safe_number(self._vars["b21"].get()), safe_number(self._vars["b22"].get())]]
        elif case["lattice"] == "triangular":
            case["direct_basis"] = [[0.5, 0.5], [3.0 ** 0.5 / 2.0, -3.0 ** 0.5 / 2.0]]
        else:
            case["direct_basis"] = [[1.0, 0.0], [0.0, 1.0]]
        representation = value("material_representation", "epsilon")
        geometry = {**case["geometry"], "epsilon_background": epsilon_from_editor(value("background_material", case["geometry"].get("epsilon_background", 7.29)), representation)}
        motifs = []
        for item in self.motif_tree.get_children():
            name, kind, radius, center, material, sides, angle = self.motif_tree.item(item, "values")
            motif = {"name": name, "kind": kind, "radius": safe_number(radius), "center": [safe_number(value) for value in json.loads(center)], "epsilon": epsilon_from_editor(material, representation), "sides": int(safe_number(sides or 8)), "angle_degrees": safe_number(angle or 0.0)}
            if kind == "polygon":
                motif["sides"] = max(3, motif["sides"])
            motifs.append(motif)
        if not motifs:
            raise ValueError("a model must retain at least one motif")
        geometry["motifs"] = motifs
        geometry["kind"] = motifs[0]["kind"]
        geometry["radius"] = motifs[0]["radius"]
        geometry["center"] = motifs[0]["center"]
        case["geometry"] = geometry
        old_affine = case.get("affine", {})
        deformation_kind = value("deformation_kind", "custom" if old_affine.get("linear") != [[1.0, 0.0], [0.0, 1.0]] else "none")
        if deformation_kind == "uniaxial":
            matrix = uniaxial_matrix(value("deformation_factor", 1.0), value("deformation_angle", 0.0)).tolist()
            factor = safe_number(value("deformation_factor", 1.0))
            angle = safe_number(value("deformation_angle", 0.0))
        elif deformation_kind == "custom":
            matrix = [[safe_number(value("a11", old_affine.get("linear", [[1.0, 0.0], [0.0, 1.0]])[0][0])), safe_number(value("a12", old_affine.get("linear", [[1.0, 0.0], [0.0, 1.0]])[0][1]))], [safe_number(value("a21", old_affine.get("linear", [[1.0, 0.0], [0.0, 1.0]])[1][0])), safe_number(value("a22", old_affine.get("linear", [[1.0, 0.0], [0.0, 1.0]])[1][1]))]]
            factor, angle = 1.0, 0.0
        else:
            matrix, factor, angle = [[1.0, 0.0], [0.0, 1.0]], 1.0, 0.0
        translation = [safe_number(value("tx", old_affine.get("translation", [0.0, 0.0])[0])), safe_number(value("ty", old_affine.get("translation", [0.0, 0.0])[1]))]
        case["deformation"] = {"kind": deformation_kind, "factor": factor, "angle_degrees": angle, "linear": matrix, "translation": translation}
        case["affine"] = {"linear": matrix, "translation": translation}
        case["basis_policy"] = "auto" if case["lattice"] in {"triangular", "square"} else "custom"
        case["name"] = model_name(case["lattice"], motifs)
        case["geometry"]["name"] = case["name"]
        return case

    def _populate_motifs(self) -> None:
        if not hasattr(self, "motif_tree"):
            return
        self.motif_tree.delete(*self.motif_tree.get_children())
        geometry = self.project["case"]["geometry"]
        motifs = geometry.get("motifs") or [geometry]
        representation = self.project.get("ui_state", {}).get("material_representation", "epsilon")
        for index, motif in enumerate(motifs):
            epsilon = motif.get("epsilon", geometry.get("epsilon_inclusion", 1.0))
            self.motif_tree.insert("", "end", iid=f"motif-{index + 1}", values=(motif.get("name", f"motif-{index + 1}"), motif.get("kind", "circle"), motif.get("radius", 0.2), json.dumps(motif.get("center", [0.5, 0.5])), editor_value(epsilon, representation), motif.get("sides", 8), motif.get("angle_degrees", 0.0)))
        selected = self.project.get("ui_state", {}).get("selected_motif")
        if selected and self.motif_tree.exists(selected):
            self.motif_tree.selection_set(selected)
        elif self.motif_tree.get_children():
            self.motif_tree.selection_set(self.motif_tree.get_children()[0])

    def _add_motif(self) -> None:
        index = len(self.motif_tree.get_children()) + 1
        representation = self._vars["material_representation"].get()
        dialog = MotifDialog(
            self,
            motif={"name": f"Circle {index}", "kind": "circle", "radius": 0.2, "center": [0.0, 0.0], "epsilon": 1.0, "sides": 6, "angle_degrees": 0.0},
            representation=representation,
            title="Add motif",
        )
        self.wait_window(dialog)
        if dialog.result is None:
            return
        item = f"motif-{index}"
        self._set_motif_row(item, dialog.result, insert=True)
        self.motif_tree.selection_set(item)
        self._mark_pending()
        self._refresh_preview()

    def _edit_motif(self) -> None:
        selection = self.motif_tree.selection()
        if not selection:
            return
        item = selection[0]
        name, kind, radius, center, material, sides, angle = self.motif_tree.item(item, "values")
        representation = self._vars["material_representation"].get()
        dialog = MotifDialog(
            self,
            motif={
                "name": name,
                "kind": kind,
                "radius": safe_number(radius),
                "center": [safe_number(value) for value in json.loads(center)],
                "epsilon": epsilon_from_editor(material, representation),
                "sides": int(safe_number(sides or 6)),
                "angle_degrees": safe_number(angle or 0.0),
            },
            representation=representation,
            title="Edit motif",
        )
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self._set_motif_row(item, dialog.result)
        self._mark_pending()
        self._refresh_preview()

    def _set_motif_row(self, item: str, motif: dict[str, Any], *, insert: bool = False) -> None:
        representation = self._vars["material_representation"].get()
        values = (
            motif["name"], motif["kind"], motif["radius"], json.dumps(motif["center"]),
            editor_value(motif["epsilon"], representation), motif.get("sides", 0), motif.get("angle_degrees", 0.0),
        )
        if insert:
            self.motif_tree.insert("", "end", iid=item, values=values)
        else:
            self.motif_tree.item(item, values=values)

    def _remove_motif(self) -> None:
        if len(self.motif_tree.get_children()) <= 1:
            messagebox.showerror("Remove motif", "A model must retain at least one motif")
            return
        for item in self.motif_tree.selection():
            self.motif_tree.delete(item)
        self._mark_pending()
        self._refresh_preview()

    def _calculation_from_controls(self) -> dict[str, Any]:
        calculation = dict(self._selected_calculation_entry()["parameters"])
        operation = self._vars["operation"].get()
        grid_size = calculation.get("grid_size", 8)
        if operation == "fields_energy":
            grid_size = int(float(self._vars["field_grid"].get()))
        elif operation == "berry":
            grid_size = int(float(self._vars["berry_grid"].get()))
        calculation.update({"operation": operation, "qpoint": [float(self._vars["qx"].get()), float(self._vars["qy"].get())], "band_one_based": int(float(self._vars["band"].get())), "composite_bands_one_based": [int(value.strip()) for value in self._vars["composite"].get().split(",") if value.strip()], "gmax": float(self._vars["gmax"].get()), "numeig": int(float(self._vars["numeig"].get())), "polarization": self._vars["polarization"].get(), "grid_size": grid_size, "efs_grid_size": int(float(self._vars["efs_grid"].get())), "berry_step": float(self._vars["berry_step"].get()), "samples_per_segment": int(float(self._vars["samples"].get())), "sampling_mode": self._vars["berry_sampling"].get(), "berry_record_path": self._vars["source_result"].get() or None, "response_weights": [float(value.strip()) for value in self._vars["response_weights"].get().split(",") if value.strip()] or None})
        return calculation

    def _sync(self) -> None:
        self.project["model"] = self._case_from_controls()
        self.project.setdefault("ui_state", {}).update({
            "material_representation": self._vars["material_representation"].get(),
            "selected_motif": (self.motif_tree.selection() or [self.project.get("ui_state", {}).get("selected_motif", "motif-1")])[0],
        })
        self._selected_calculation_entry()["parameters"] = self._calculation_from_controls()
        self._selected_calculation_entry()["operation"] = self._selected_calculation_entry()["parameters"]["operation"]
        self.project["plot"].update({"width_px": int(float(self._vars["width_px"].get())), "height_px": int(float(self._vars["height_px"].get())), "dpi": int(float(self._vars["dpi"].get())), "title": self._vars["title"].get(), "x_label": self._vars["x_label"].get(), "y_label": self._vars["y_label"].get(), "linewidth": float(self._vars["linewidth"].get()), "marker_size": float(self._vars["marker_size"].get()), "cmap": self._vars["cmap"].get(), "component_index": int(float(self._vars["component_index"].get())), "field_quantity": self._vars["field_quantity"].get(), "berry_coloring": bool(self._vars["berry_coloring"].get()), "berry_interpolation": bool(self._vars["berry_interpolation"].get()), "colorbar": bool(self._vars["colorbar"].get())})
        for name in ("grid", "legend", "band_line", "band_markers"):
            if name in self._vars:
                self.project["plot"][name] = self._vars[name].get() in {True, "True", "1"}
        if not self.project["plot"].get("band_line") and not self.project["plot"].get("band_markers"):
            raise ValueError("enable Band lines, Band markers, or both")
        for name in ("x_limits", "y_limits"):
            value = self._vars[name].get().strip()
            self.project["plot"][name] = [float(item.strip()) for item in value.split(",")] if value else None
        self.dirty = True
        self._update_title()

    def _refresh_preview(self) -> None:
        try:
            case = self._case_from_controls()
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.patches import Circle, Ellipse, Polygon
            if not hasattr(self, "geometry_canvases"):
                self.geometry_canvases = {}
                self.geometry_toolbars = {}
            for canvas in self.geometry_canvases.values():
                canvas.get_tk_widget().destroy()
            for toolbar in self.geometry_toolbars.values():
                toolbar.destroy()
            self.geometry_canvases.clear()
            self.geometry_toolbars.clear()
            view_map = {"Motif": "motif", "Unit Cell": "unit_cell", "Motif Array": "motif_array", "Lattice Sites": "lattice_sites", "Reciprocal & BZ": "reciprocal_bz", "Epsilon": "epsilon"}

            def add_shape(axis, shape, shift=(0.0, 0.0), color="tab:blue", alpha=0.45):
                offset = np.asarray(shift, dtype=float)
                if shape["kind"] == "circle":
                    center = np.asarray(shape["center"]) + offset
                    patch = Circle(center, shape["radius"], facecolor=color, edgecolor="black", alpha=alpha)
                elif shape["kind"] == "ellipse":
                    center = np.asarray(shape["center"]) + offset
                    patch = Ellipse(center, shape["width"], shape["height"], angle=shape["angle_degrees"], facecolor=color, edgecolor="black", alpha=alpha)
                else:
                    patch = Polygon(np.asarray(shape["vertices"]) + offset, closed=True, facecolor=color, edgecolor="black", alpha=alpha)
                axis.add_patch(patch)

            def shape_points(shape):
                if shape["kind"] == "circle":
                    center = np.asarray(shape["center"])
                    radius = shape["radius"]
                    return np.asarray([center - radius, center + radius])
                if shape["kind"] == "ellipse":
                    center = np.asarray(shape["center"])
                    radius = max(shape["width"], shape["height"]) / 2.0
                    return np.asarray([center - radius, center + radius])
                return np.asarray(shape["vertices"])

            for tab_name, view in view_map.items():
                data = preview_geometry(case, view=view, size=96)
                figure = Figure(figsize=(4.5, 3.5), dpi=100)
                axis = figure.add_subplot(111)
                if view == "motif":
                    shapes = data.get("motifs", [])
                    selected = self.motif_tree.selection() if self.motif_tree.selection() else (self.project.get("ui_state", {}).get("selected_motif", "motif-1"),)
                    selected_index = max(0, int(str(selected[0]).rsplit("-", 1)[-1]) - 1) if selected else 0
                    if shapes:
                        shape = shapes[min(selected_index, len(shapes) - 1)]
                        add_shape(axis, shape, color="tab:blue", alpha=0.65)
                        bounds = shape_points(shape)
                        center = bounds.mean(axis=0)
                        half = max(float(np.ptp(bounds[:, 0])), float(np.ptp(bounds[:, 1])), 0.1) * 0.75
                        axis.set_xlim(center[0] - half, center[0] + half)
                        axis.set_ylim(center[1] - half, center[1] + half)
                elif view == "unit_cell":
                    basis = np.asarray(data["direct_basis"])
                    cell = np.asarray([[0, 0], basis[:, 0], basis[:, 0] + basis[:, 1], basis[:, 1], [0, 0]])
                    axis.plot(cell[:, 0], cell[:, 1], color="black", label="unit cell")
                    axis.arrow(0, 0, basis[0, 0], basis[1, 0], color="tab:red", length_includes_head=True, head_width=0.03)
                    axis.arrow(0, 0, basis[0, 1], basis[1, 1], color="tab:blue", length_includes_head=True, head_width=0.03)
                    axis.annotate("a1", basis[:, 0], color="tab:red")
                    axis.annotate("a2", basis[:, 1], color="tab:blue")
                    for index, shape in enumerate(data.get("motifs", [])):
                        add_shape(axis, shape, color=f"C{index % 10}", alpha=0.5)
                    axis.legend()
                elif view == "motif_array":
                    for n1 in range(-1, 2):
                        for n2 in range(-1, 2):
                            shift = data["direct_basis"] @ np.array([n1, n2], dtype=float)
                            for index, shape in enumerate(data.get("motifs", [])):
                                add_shape(axis, shape, shift=shift, color=f"C{index % 10}", alpha=0.4)
                elif view == "reciprocal_bz":
                    vertices = data["bz_vertices"]
                    axis.plot(*np.vstack((vertices, vertices[0])).T, color="black", label="first BZ")
                    sites = data["points"]
                    axis.scatter(sites[:, 0], sites[:, 1], s=12, label="reciprocal sites")
                    reciprocal = data["reciprocal_basis"]
                    axis.arrow(0, 0, reciprocal[0, 0], reciprocal[1, 0], color="tab:red", length_includes_head=True, head_width=0.08)
                    axis.arrow(0, 0, reciprocal[0, 1], reciprocal[1, 1], color="tab:blue", length_includes_head=True, head_width=0.08)
                    axis.annotate("b1", reciprocal[:, 0], color="tab:red")
                    axis.annotate("b2", reciprocal[:, 1], color="tab:blue")
                    axis.legend()
                elif view == "lattice_sites":
                    points = data["points"]
                    axis.scatter(points[:, 0], points[:, 1], s=18)
                elif data.get("epsilon") is not None:
                    image = axis.imshow(data["epsilon"], origin="lower", aspect="equal", cmap="viridis")
                    if view == "epsilon":
                        figure.colorbar(image, ax=axis, label="epsilon")
                title = tab_name
                if view == "reciprocal_bz":
                    title += f" — max |AᵀB−2πI|={data['reciprocal_identity_residual']:.2e}"
                axis.set_title(title)
                axis.set_aspect("equal", adjustable="box")
                axis.set_box_aspect(1)
                host = self.geometry_hosts[tab_name]
                canvas = FigureCanvasTkAgg(figure, master=host)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                toolbar = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(side="bottom", fill="x")
                self.geometry_canvases[tab_name] = canvas
                self.geometry_toolbars[tab_name] = toolbar
            self.preview_figure = self.geometry_canvases["Motif"].figure
            self.preview_canvas = self.geometry_canvases["Motif"]
            self.preview_toolbar = self.geometry_toolbars["Motif"]
            self.preview_status.configure(text="Preview current; pending Run")
        except Exception as exc:
            self.preview_status.configure(text=f"Preview unavailable: {exc}")

    def _new(self) -> None:
        if not self._confirm_discard():
            return
        self.project = new_project()
        self.project_path = None
        self.dirty = True
        self._populate()

    def _open(self) -> None:
        initialdir, initialfile = _dialog_location(self, "project")
        path = filedialog.askopenfilename(initialdir=initialdir, initialfile=initialfile, filetypes=[("LegumePhC Studio project", f"*{PROJECT_SUFFIX}"), ("JSON", "*.json")])
        if path:
            try:
                self.project = load_project(path)
                self.project_path = Path(path).resolve()
                self.dirty = False
                self._populate()
            except Exception as exc:
                messagebox.showerror("Open project", str(exc))

    def _save(self) -> bool:
        if self.project_path is None:
            return self._save_as()
        try:
            self._sync()
            save_project(self.project_path, self.project)
            self.dirty = False
            self._update_title()
            return True
        except Exception as exc:
            messagebox.showerror("Save project", str(exc))
            return False

    def _save_as(self) -> bool:
        initialdir, initialfile = _dialog_location(self, "project", saving=True)
        path = filedialog.asksaveasfilename(initialdir=initialdir, initialfile=initialfile, defaultextension=PROJECT_SUFFIX, filetypes=[("LegumePhC Studio project", f"*{PROJECT_SUFFIX}")])
        if not path:
            return False
        target = Path(path).resolve()
        if self.project_path is not None and self.project.get("records") and target.parent != self.project_path.parent:
            messagebox.showerror("Save As", "Cannot relocate a project with records to another directory; save in the current directory or start a new project.")
            return False
        self.project_path = target
        return self._save()

    def _load_preset(self) -> None:
        initialdir, initialfile = _dialog_location(self, "preset")
        path = filedialog.askopenfilename(initialdir=initialdir, initialfile=initialfile, filetypes=[("LegumePhC parameter preset", "*.legumephc-preset.json"), ("JSON", "*.json")])
        if path:
            try:
                self.project = apply_preset(self.project, load_preset(path))
                self.dirty = True
                self._populate()
            except Exception as exc:
                messagebox.showerror("Load preset", str(exc))

    def _save_preset(self) -> None:
        initialdir, initialfile = _dialog_location(self, "preset", saving=True)
        path = filedialog.asksaveasfilename(initialdir=initialdir, initialfile=initialfile, defaultextension=".legumephc-preset.json", filetypes=[("LegumePhC parameter preset", "*.legumephc-preset.json")])
        if not path:
            return
        try:
            self._sync()
            preset = new_preset(Path(path).stem.replace(".legumephc-preset", ""))
            preset["parameters"] = {"case": self.project["case"], "calculation": self.project["calculation"], "ui_state": self.project.get("ui_state", {})}
            save_preset(path, preset)
            self._append_log(f"parameter preset saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save preset", str(exc))

    def _run(self) -> None:
        if self.worker is not None and self.worker.poll() is None:
            return
        try:
            self._sync()
            validate_project(self.project)
            if self.project_path is None and not self._save_as():
                self.calc_status.configure(text="Run cancelled: save the project first")
                return
            request = build_worker_request(self.project, self.project_path)
            self.worker = start_worker(request, python_executable=sys.executable, cwd=ROOT)
            # The UI remains interactive while the worker runs.  Freeze the
            # exact request so completion cannot be rebound to a calculation
            # that the user selected or edited in the meantime.
            self._active_request = json.loads(json.dumps(request))
            self._run_started_at = time.monotonic()
            self.progress_bar.stop()
            self.progress_bar.configure(mode="indeterminate", value=0)
            self.progress_bar.start(12)
            calculation = request["calculation"]
            self.calc_status.configure(text=f"Starting {calculation['operation']}…")
            self.calc_detail.configure(text=f"gmax={calculation.get('gmax')} · bands={calculation.get('composite_bands_one_based')} · elapsed 0.0 s")
            self.run_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
            self._append_log(f"started {calculation['operation']}")
            self.after(100, self._poll_worker)
        except Exception as exc:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate", value=0)
            self.calc_status.configure(text=f"Validation failed: {exc}")
            self.calc_detail.configure(text="No worker started; correct the calculation settings and run again")

    def _poll_worker(self) -> None:
        if self.worker is None:
            return
        for event in self.worker.read_events():
            self._handle_worker_event(event)
        if self.worker.poll() is None:
            if self._run_started_at is not None:
                elapsed = time.monotonic() - self._run_started_at
                base = self.calc_detail.cget("text").split(" · elapsed", 1)[0]
                self.calc_detail.configure(text=f"{base} · elapsed {elapsed:.1f} s")
            self.after(100, self._poll_worker)
            return
        stdout, stderr = self.worker.communicate()
        if stderr:
            self._append_log(stderr.strip())
        result: dict[str, Any] = {}
        try:
            messages = [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]
            result = next((item for item in reversed(messages) if item.get("event") in {"completed", "failed"}), messages[-1] if messages else {})
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "worker failed"))
            project_dir = self.project_path.parent if self.project_path else ROOT
            reference = record_reference(Path(result["record_path"]), project_dir)
            active_request = self._active_request or {}
            calculation_id = active_request.get("calculation_id")
            calculation = next((item for item in self.project.get("calculations", []) if item["id"] == calculation_id), None)
            if calculation is None:
                raise RuntimeError("the calculation used for this run no longer exists")
            result_id = f"result-{len(self.project.get('results', [])) + 1}"
            self.project["results"].append({"id": result_id, "calculation_id": calculation["id"], "record_reference": reference, "model_snapshot": active_request["model_snapshot"], "calculation_snapshot": active_request["calculation_snapshot"], "plot": json.loads(json.dumps(self.project["plot"]))})
            self.project["selected_result"] = reference["path"]
            self.dirty = True
            self._populate_results()
            self.calc_status.configure(text=f"Completed: {reference['path']}")
            self.calc_detail.configure(text=f"{result['metadata']['operation']} · record ready · elapsed {(time.monotonic() - self._run_started_at):.1f} s")
            self._append_log(f"completed {result['metadata']['operation']}: {reference['path']}")
            if self._plot_after_var.get():
                self._plot_selected()
        except Exception as exc:
            self.calc_status.configure(text=f"Worker failed: {exc}")
            self._append_log(f"worker failed: {exc}")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=100 if result.get("ok") else 0)
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.worker = None
        self._active_request = None
        self._run_started_at = None

    def _handle_worker_event(self, event: dict[str, Any]) -> None:
        phase = str(event.get("phase") or event.get("event", "working")).replace("_", " ")
        message = str(event.get("message") or phase)
        completed, total = event.get("completed"), event.get("total")
        if completed is not None and total:
            percent = max(0.0, min(100.0, 100.0 * float(completed) / float(total)))
            if str(self.progress_bar.cget("mode")) != "determinate":
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
            self.progress_bar.configure(value=percent)
            self.calc_status.configure(text=f"{phase.title()}: {completed}/{total} ({percent:.0f}%)")
        else:
            if str(self.progress_bar.cget("mode")) != "indeterminate":
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start(12)
            self.calc_status.configure(text=message)
        self.calc_detail.configure(text=message)
        if event.get("event") in {"phase", "record_written"}:
            self._append_log(message)

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.worker = None
            self._active_request = None
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate", value=0)
            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.calc_status.configure(text="Cancelled; no child remains")
            self.calc_detail.configure(text="Exact child process terminated")
            self._append_log("worker cancelled")
            self._run_started_at = None

    def _populate_results(self) -> None:
        if not hasattr(self, "result_list"):
            return
        self.result_list.delete(0, tk.END)
        project_dir = self.project_path.parent if self.project_path else ROOT
        for reference in self.project.get("records", []):
            available, reason = record_available(project_dir, reference)
            operation = reference.get("identity", {}).get("operation", "unknown")
            label = f"{'AVAILABLE' if available else 'UNAVAILABLE'} [{operation}]: {reference.get('path', '?')}"
            self.result_list.insert(tk.END, label)
            reference["_availability"] = reason

    def _select_result(self) -> None:
        selected = self.result_list.curselection()
        if not selected:
            return
        reference = self.project["records"][selected[0]]
        self.project["selected_result"] = reference["path"]
        self.result_status.configure(text=reference.get("_availability", "selected"))
        self._update_style_visibility(self._record_operation(reference))

    def _record_operation(self, reference: dict[str, Any]) -> str:
        project_dir = self.project_path.parent if self.project_path else ROOT
        try:
            config = json.loads((project_dir / reference["path"] / "config.json").read_text(encoding="utf-8"))
            return config.get("identity", {}).get("operation", "result")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return reference.get("identity", {}).get("operation", "result")

    def _plot_selected(self) -> None:
        self._select_result()
        selected = self.project.get("selected_result")
        if not selected:
            return
        project_dir = self.project_path.parent if self.project_path else ROOT
        reference = next((item for item in self.project["records"] if item.get("path") == selected), None)
        if reference is None or not record_available(project_dir, reference)[0]:
            self.result_status.configure(text="Selected record is unavailable or has a mismatched identity")
            return
        self.figure = plot_record(project_dir / selected, self.project["plot"])
        if self.plot_host is not None:
            self.plot_host.destroy()
        self.plot_host = ttk.Frame(self.result_side)
        self.plot_host.pack(fill="both", expand=True)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_host)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_host, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")
        if hasattr(self, "style_status"):
            self.style_status.configure(text="Style current")

    def _apply_plot_style(self) -> None:
        try:
            self._sync()
            self._plot_selected()
            if self.figure is None:
                self.style_status.configure(text="Select a result before applying style")
        except Exception as exc:
            self.style_status.configure(text=f"Plot style unavailable: {exc}")

    def _export(self) -> None:
        if self.figure is None:
            return
        initialdir, initialfile = _dialog_location(self, "export", saving=True, record_path=self.project.get("selected_result"))
        path = filedialog.asksaveasfilename(initialdir=initialdir, initialfile=initialfile, defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            export_figure(self.figure, path, width_px=int(float(self._vars["width_px"].get())), height_px=int(float(self._vars["height_px"].get())), dpi=int(float(self._vars["dpi"].get())))

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved project", "Save changes to the current project?")
        if answer is None:
            return False
        return self._save() if answer else True

    def _close(self) -> None:
        if not self._confirm_discard():
            return
        if self.worker is not None:
            self.worker.cancel()
            self.worker = None
            self._active_request = None
        self.destroy()

    def _update_title(self) -> None:
        label = self.project_path.name if self.project_path else self.project.get("name", "Untitled")
        self.title(f"LegumePhC Studio — {label}{'*' if self.dirty else ''}")


def main() -> int:
    StudioApp().mainloop()
    return 0
