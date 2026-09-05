from __future__ import annotations

import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .plotting import default_plot_style, export_figure, plot_record
from .preview import preview_geometry
from .profile import model_from_case
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


class StudioApp(tk.Tk):
    """Small single-window Studio; all formal solves run in one child."""

    def __init__(self, project: dict[str, Any] | None = None, project_path: str | Path | None = None):
        super().__init__()
        self.project = project or new_project()
        self.project_path = Path(project_path).resolve() if project_path else None
        self.dirty = False
        self.worker: WorkerProcess | None = None
        self.figure = None
        self.canvas = None
        self.preview_figure = None
        self.preview_canvas = None
        self.toolbar = None
        self.plot_host = None
        self._vars: dict[str, tk.Variable] = {}
        self._build()
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

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self.geometry_tab = ttk.Frame(notebook)
        self.calculation_tab = ttk.Frame(notebook)
        self.results_tab = ttk.Frame(notebook)
        self.style_tab = ttk.Frame(notebook)
        notebook.add(self.geometry_tab, text="Geometry")
        notebook.add(self.calculation_tab, text="Calculation")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.style_tab, text="Plot Style")
        self._build_geometry_tab()
        self._build_calculation_tab()
        self._build_results_tab()
        self._build_style_tab()
        self.log = tk.Text(self, height=5, state="disabled")
        self.log.pack(fill="x", padx=6, pady=(0, 6))
        self._loading = False
        for name in ("lattice", "lattice_constant", "kind", "radius", "sides", "angle", "b11", "b12", "b21", "b22", "a11", "a12", "a21", "a22", "tx", "ty"):
            self._vars[name].trace_add("write", self._mark_pending)

    def _var(self, name: str, value: Any = "") -> tk.Variable:
        variable = tk.StringVar(self, str(value))
        self._vars[name] = variable
        return variable

    def _label_entry(self, parent, row: int, label: str, name: str, width: int = 12) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self._var(name), width=width).grid(row=row, column=1, sticky="w", padx=4, pady=2)

    def _build_geometry_tab(self) -> None:
        controls = ttk.LabelFrame(self.geometry_tab, text="Model parameters")
        controls.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Label(controls, text="Lattice").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.lattice_menu = ttk.Combobox(controls, textvariable=self._var("lattice"), values=("triangular", "square", "custom"), state="readonly", width=13)
        self.lattice_menu.grid(row=0, column=1, padx=4, pady=2)
        self._label_entry(controls, 1, "Lattice constant", "lattice_constant")
        ttk.Label(controls, text="Motif").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.kind_menu = ttk.Combobox(controls, textvariable=self._var("kind"), values=("circle", "polygon"), state="readonly", width=13)
        self.kind_menu.grid(row=2, column=1, padx=4, pady=2)
        self._label_entry(controls, 3, "Radius", "radius")
        self._label_entry(controls, 4, "Polygon sides", "sides")
        self._label_entry(controls, 5, "Angle degrees", "angle")
        ttk.Label(controls, text="Custom direct basis").grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 2))
        for row, left, right in ((7, "b11", "b12"), (8, "b21", "b22")):
            ttk.Entry(controls, textvariable=self._var(left), width=8).grid(row=row, column=0, padx=4, pady=2)
            ttk.Entry(controls, textvariable=self._var(right), width=8).grid(row=row, column=1, padx=4, pady=2)
        ttk.Label(controls, text="Affine matrix").grid(row=9, column=0, columnspan=2, sticky="w", padx=4, pady=(8, 2))
        for row, name in enumerate(("a11", "a12", "a21", "a22"), start=10):
            ttk.Entry(controls, textvariable=self._var(name), width=8).grid(row=10 + (row - 10) // 2, column=(row - 10) % 2, padx=4, pady=2)
        self._label_entry(controls, 12, "Translation x", "tx")
        self._label_entry(controls, 13, "Translation y", "ty")
        ttk.Button(controls, text="Refresh preview", command=self._refresh_preview).grid(row=14, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        ttk.Label(controls, text="Preview").grid(row=15, column=0, sticky="w", padx=4)
        self.preview_view = ttk.Combobox(controls, textvariable=self._var("preview_view"), values=("unit_cell", "motif_array", "lattice_sites", "epsilon"), state="readonly", width=13)
        self.preview_view.grid(row=15, column=1, padx=4, pady=2)
        self.preview_status = ttk.Label(controls, text="Pending Refresh")
        self.preview_status.grid(row=16, column=0, columnspan=2, sticky="w", padx=4)
        self.preview_host = ttk.Frame(self.geometry_tab)
        self.preview_host.pack(side="left", fill="both", expand=True, padx=6, pady=6)

    def _build_calculation_tab(self) -> None:
        controls = ttk.LabelFrame(self.calculation_tab, text="Calculation settings")
        controls.pack(anchor="nw", padx=6, pady=6)
        ttk.Label(controls, text="Calculation").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.operation_menu = ttk.Combobox(controls, textvariable=self._var("operation"), values=("frequency_at_k", "band_structure", "fields_energy", "efs", "berry", "berry_curvature_dipole"), state="readonly", width=24)
        self.operation_menu.grid(row=0, column=1, padx=4, pady=2)
        for row, label, name in ((1, "q x", "qx"), (2, "q y", "qy"), (3, "Band (one-based)", "band"), (4, "Composite bands (one-based)", "composite"), (5, "gmax", "gmax"), (6, "numeig", "numeig"), (7, "Grid size", "grid"), (8, "EFS grid", "efs_grid"), (9, "Berry step", "berry_step")):
            self._label_entry(controls, row, label, name, 24 if name == "composite" else 12)
        ttk.Label(controls, text="Polarization").grid(row=10, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(controls, textvariable=self._var("polarization"), values=("te", "tm"), state="readonly", width=12).grid(row=10, column=1, sticky="w", padx=4, pady=2)
        buttons = ttk.Frame(controls)
        buttons.grid(row=11, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        ttk.Button(buttons, text="Run", command=self._run).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self._cancel).pack(side="left", padx=4)
        self.plot_after = ttk.Checkbutton(buttons, text="Plot after Run", variable=self._plot_after_var)
        self.plot_after.pack(side="left")
        self.calc_status = ttk.Label(controls, text="Ready")
        self.calc_status.grid(row=12, column=0, columnspan=2, sticky="w", padx=4)

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
        ttk.Button(controls, text="Export current plot…", command=self._export).grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=6)

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
        values = {"lattice": case["lattice"], "lattice_constant": case.get("lattice_constant", 1.0), "kind": geometry["kind"], "radius": geometry["radius"], "sides": geometry.get("sides", 16), "angle": geometry.get("angle_degrees", 0.0), "b11": basis[0][0], "b12": basis[0][1], "b21": basis[1][0], "b22": basis[1][1], "a11": case["affine"]["linear"][0][0], "a12": case["affine"]["linear"][0][1], "a21": case["affine"]["linear"][1][0], "a22": case["affine"]["linear"][1][1], "tx": case["affine"]["translation"][0], "ty": case["affine"]["translation"][1], "preview_view": "epsilon"}
        calculation = self.project["calculation"]
        values.update({"operation": calculation["operation"], "qx": calculation["qpoint"][0], "qy": calculation["qpoint"][1], "band": calculation["band_one_based"], "composite": ",".join(map(str, calculation["composite_bands_one_based"])), "gmax": calculation["gmax"], "numeig": calculation["numeig"], "polarization": calculation["polarization"], "grid": calculation["grid_size"], "efs_grid": calculation["efs_grid_size"], "berry_step": calculation["berry_step"]})
        values.update(self.project["plot"])
        for name, value in values.items():
            if name in self._vars:
                self._vars[name].set(value)
        self._refresh_preview()
        self._populate_results()
        self._loading = False

    def _mark_pending(self, *_args) -> None:
        if getattr(self, "_loading", True):
            return
        self.dirty = True
        self.preview_status.configure(text="Pending Refresh/Run")
        self.calc_status.configure(text="Pending Run")
        self._update_title()

    def _case_from_controls(self) -> dict[str, Any]:
        case = dict(self.project["case"])
        case["lattice"] = self._vars["lattice"].get()
        case["lattice_constant"] = float(self._vars["lattice_constant"].get())
        case["direct_basis"] = [[float(self._vars["b11"].get()), float(self._vars["b12"].get())], [float(self._vars["b21"].get()), float(self._vars["b22"].get())]]
        case["geometry"] = {**case["geometry"], "kind": self._vars["kind"].get(), "radius": float(self._vars["radius"].get()), "sides": int(float(self._vars["sides"].get())), "angle_degrees": float(self._vars["angle"].get())}
        case["affine"] = {"linear": [[float(self._vars["a11"].get()), float(self._vars["a12"].get())], [float(self._vars["a21"].get()), float(self._vars["a22"].get())]], "translation": [float(self._vars["tx"].get()), float(self._vars["ty"].get())]}
        return case

    def _calculation_from_controls(self) -> dict[str, Any]:
        calculation = dict(self.project["calculation"])
        calculation.update({"operation": self._vars["operation"].get(), "qpoint": [float(self._vars["qx"].get()), float(self._vars["qy"].get())], "band_one_based": int(float(self._vars["band"].get())), "composite_bands_one_based": [int(value.strip()) for value in self._vars["composite"].get().split(",") if value.strip()], "gmax": float(self._vars["gmax"].get()), "numeig": int(float(self._vars["numeig"].get())), "polarization": self._vars["polarization"].get(), "grid_size": int(float(self._vars["grid"].get())), "efs_grid_size": int(float(self._vars["efs_grid"].get())), "berry_step": float(self._vars["berry_step"].get())})
        return calculation

    def _sync(self) -> None:
        self.project["case"] = self._case_from_controls()
        self.project["calculation"] = self._calculation_from_controls()
        self.project["plot"].update({name: self._vars[name].get() for name in ("width_px", "height_px", "dpi", "title", "x_label", "y_label")})
        self.project["plot"].update({name: self._vars[name].get() for name in ("grid", "legend", "band_style", "berry_coloring") if name in self._vars})
        self.dirty = True
        self._update_title()

    def _refresh_preview(self) -> None:
        try:
            case = self._case_from_controls()
            data = preview_geometry(case, view=self._vars.get("preview_view", tk.StringVar(value="epsilon")).get() or "epsilon", size=96)
            from matplotlib.figure import Figure
            self.preview_figure = Figure(figsize=(5, 4), dpi=100)
            axis = self.preview_figure.add_subplot(111)
            if data["epsilon"] is not None:
                axis.imshow(data["epsilon"], origin="lower", aspect="equal")
            else:
                points = data["points"]
                axis.scatter(points[:, 0], points[:, 1])
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(str(data["view"]))
            if self.preview_canvas is not None:
                self.preview_canvas.get_tk_widget().destroy()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=self.preview_host)
            self.preview_canvas.draw()
            self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
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
        path = filedialog.askopenfilename(filetypes=[("LegumePhC Studio project", f"*{PROJECT_SUFFIX}"), ("JSON", "*.json")])
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
        path = filedialog.asksaveasfilename(defaultextension=PROJECT_SUFFIX, filetypes=[("LegumePhC Studio project", f"*{PROJECT_SUFFIX}")])
        if not path:
            return False
        target = Path(path).resolve()
        if self.project_path is not None and self.project.get("records") and target.parent != self.project_path.parent:
            messagebox.showerror("Save As", "Cannot relocate a project with records to another directory; save in the current directory or start a new project.")
            return False
        self.project_path = target
        return self._save()

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("LegumePhC parameter preset", "*.legumephc-preset.json"), ("JSON", "*.json")])
        if path:
            try:
                self.project = apply_preset(self.project, load_preset(path))
                self.dirty = True
                self._populate()
            except Exception as exc:
                messagebox.showerror("Load preset", str(exc))

    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".legumephc-preset.json", filetypes=[("LegumePhC parameter preset", "*.legumephc-preset.json")])
        if not path:
            return
        try:
            self._sync()
            preset = new_preset(Path(path).stem.replace(".legumephc-preset", ""))
            preset["parameters"] = {"case": self.project["case"], "calculation": self.project["calculation"]}
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
            self.calc_status.configure(text="Running in exact child process…")
            self._append_log("worker started")
            self.after(100, self._poll_worker)
        except Exception as exc:
            self.calc_status.configure(text=f"Validation failed: {exc}")

    def _poll_worker(self) -> None:
        if self.worker is None:
            return
        if self.worker.poll() is None:
            self.after(100, self._poll_worker)
            return
        stdout, stderr = self.worker.communicate()
        if stderr:
            self._append_log(stderr.strip())
        try:
            result = json.loads(stdout.strip().splitlines()[-1])
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "worker failed"))
            project_dir = self.project_path.parent if self.project_path else ROOT
            reference = record_reference(Path(result["record_path"]), project_dir)
            self.project["records"].append(reference)
            self.project["selected_result"] = reference["path"]
            self.dirty = True
            self._populate_results()
            self.calc_status.configure(text=f"Completed: {reference['path']}")
            self._append_log(stdout.strip())
            if self._plot_after_var.get():
                self._plot_selected()
        except Exception as exc:
            self.calc_status.configure(text=f"Worker failed: {exc}")
            self._append_log(stdout.strip())
        self.worker = None

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.worker = None
            self.calc_status.configure(text="Cancelled; no child remains")
            self._append_log("worker cancelled")

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
        self.plot_host = ttk.Frame(self.results_tab)
        self.plot_host.pack(side="bottom", fill="both", expand=True)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_host)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_host, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")

    def _export(self) -> None:
        if self.figure is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
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
        self.destroy()

    def _update_title(self) -> None:
        label = self.project_path.name if self.project_path else self.project.get("name", "Untitled")
        self.title(f"LegumePhC Studio — {label}{'*' if self.dirty else ''}")


def main() -> int:
    StudioApp().mainloop()
    return 0
