"""Small public-API runner used by PyCharm's direct study configurations."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legumephc import (  # noqa: E402
    Affine2D,
    GeometrySpec,
    Lattice2D,
    Model2D,
    compute_berry_dipole,
    compute_field_observables,
    frequency_at_k,
    solve_bands,
    solve_berry,
    solve_efs,
)
from legumephc.berry import first_bz_plaquettes  # noqa: E402
from legumephc.studio.plotting import export_figure, plot_record  # noqa: E402
from studies.defaults import (  # noqa: E402
    BCDParameters,
    BandParameters,
    BerryParameters,
    CaseParameters,
    EFSParameters,
    FieldsParameters,
    FrequencyParameters,
    PlotParameters,
    as_dict,
)


def model_from_parameters(parameters: CaseParameters) -> Model2D:
    if parameters.lattice == "triangular":
        lattice = Lattice2D.triangular(parameters.lattice_constant)
    elif parameters.lattice == "square":
        lattice = Lattice2D.square(parameters.lattice_constant)
    else:
        if parameters.direct_basis is None:
            raise ValueError("custom cases require direct_basis")
        lattice = Lattice2D(np.asarray(parameters.direct_basis, dtype=float), kind="custom")
    sides = tuple(0 if motif.sides is None else int(motif.sides) for motif in parameters.motifs)
    geometry = GeometrySpec(
        name=parameters.name or ("Direct" + parameters.lattice.title()),
        kind=parameters.motifs[0].kind,
        radii=tuple(float(motif.radius) for motif in parameters.motifs),
        sides=sides,
        angles_degrees=tuple(float(motif.angle_degrees) for motif in parameters.motifs),
        strict_c3=False,
        epsilon_background=float(parameters.material.background_epsilon),
        epsilon_inclusion=float(parameters.material.inclusion_epsilon),
        motif_kinds=tuple(motif.kind for motif in parameters.motifs),
        motif_epsilons=tuple(float(parameters.material.inclusion_epsilon if motif.epsilon is None else motif.epsilon) for motif in parameters.motifs),
        direct_basis=lattice.direct_basis,
        centers=np.asarray([motif.center for motif in parameters.motifs], dtype=float),
    )
    affine = Affine2D(np.asarray(parameters.affine.linear, dtype=float), np.asarray(parameters.affine.translation, dtype=float))
    return Model2D(geometry, lattice, affine=affine, basis_policy=parameters.basis_policy)


def _plaquettes(model: Model2D, parameters: BerryParameters) -> np.ndarray:
    if parameters.sampling_mode == "single_plaquette":
        centers = np.asarray([parameters.center], dtype=float)
    elif parameters.sampling_mode == "explicit_centers":
        centers = np.asarray(parameters.centers, dtype=float)
    elif parameters.sampling_mode == "first_bz_grid":
        return first_bz_plaquettes(model.effective_lattice, grid_size=parameters.grid_size, step=parameters.step)
    else:
        raise ValueError(f"unsupported Berry sampling mode: {parameters.sampling_mode}")
    offsets = np.asarray([[-parameters.step, -parameters.step], [-parameters.step, parameters.step], [parameters.step, parameters.step], [parameters.step, -parameters.step]])
    return centers[:, None, :] + offsets[None, :, :]


def _record_dirs(root: Path) -> set[Path]:
    return {path for path in root.iterdir() if path.is_dir() and (path / "config.json").exists()} if root.exists() else set()


def _single_new_record(root: Path, before: set[Path]) -> Path:
    created = sorted(_record_dirs(root) - before)
    if len(created) != 1:
        raise RuntimeError(f"expected exactly one immutable record, found {len(created)}")
    return created[0]


def show_figure(figure) -> None:
    """Display a returned Figure in a real, closeable Tk window."""
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    window = tk.Tk()
    window.title(figure.axes[0].get_title() if figure.axes else "LegumePhC result")
    canvas = FigureCanvasTkAgg(figure, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas, window).update()
    window.protocol("WM_DELETE_WINDOW", window.destroy)
    window.mainloop()


def run_operation(case_name: str, case: CaseParameters, parameters: Any, *, record_root: str | Path | None = None, plot: PlotParameters | None = None, show: bool = True, testing: bool = False) -> dict[str, Any]:
    """Run one operation and attach one default figure to its immutable record."""
    model = model_from_parameters(case)
    root = Path(record_root) if record_root is not None else ROOT / "studies" / case_name / "results"
    root.mkdir(parents=True, exist_ok=True)
    before = _record_dirs(root)
    solver = parameters.solver if hasattr(parameters, "solver") else None
    operation = type(parameters).__name__
    if isinstance(parameters, FrequencyParameters):
        value = frequency_at_k(model, parameters.qpoint, gmax=solver.gmax, band=parameters.band_one_based - 1, pol=solver.polarization, record_root=root)
        summary = {"frequency": value, "band_one_based": parameters.band_one_based}
        operation = "frequency_at_k"
    elif isinstance(parameters, BandParameters):
        result = solve_bands(model, path=parameters.path, gmax=solver.gmax, numeig=solver.numeig, pol=solver.polarization, samples_per_segment=parameters.samples_per_segment, record_root=root)
        summary = {"qpoint_count": len(result["qpoints"]), "path_labels": result["path_labels"]}
        operation = "solve_bands"
    elif isinstance(parameters, FieldsParameters):
        bands = tuple(value - 1 for value in parameters.bands_one_based)
        solved = solve_bands(model, np.asarray([parameters.qpoint]), gmax=solver.gmax, numeig=solver.numeig, pol=solver.polarization)
        result = compute_field_observables(solved, model, bands=bands, grid_size=parameters.grid_size, record_root=root)
        summary = {"grid_size": parameters.grid_size, "bands_one_based": parameters.bands_one_based}
        operation = "compute_field_observables"
    elif isinstance(parameters, EFSParameters):
        result = solve_efs(model, gmax=solver.gmax, grid_size=parameters.grid_size, bands=tuple(value - 1 for value in parameters.bands_one_based), numeig=solver.numeig, pol=solver.polarization, record_root=root)
        summary = {"grid_shape": result["grid_shape"], "sample_count": len(result["qpoints"])}
        operation = "solve_efs"
    elif isinstance(parameters, BerryParameters):
        result = solve_berry(model, _plaquettes(model, parameters), gmax=solver.gmax, bands=tuple(value - 1 for value in parameters.bands_one_based), rank=parameters.rank, numeig=solver.numeig, pol=solver.polarization, convergence_status=parameters.convergence_status, record_root=root)
        summary = {"plaquette_count": result["qualification"]["plaquette_count"], "qualification": result["qualification"]}
        operation = "solve_berry"
    elif isinstance(parameters, BCDParameters):
        if not parameters.berry_record_path:
            raise ValueError("BCD requires an explicit qualified Berry record path; no synthetic curvature is generated")
        berry_path = Path(parameters.berry_record_path)
        source_config = json.loads((berry_path / "config.json").read_text(encoding="utf-8"))
        if source_config.get("identity", {}).get("model") != "Model2D" or source_config.get("identity", {}).get("operation") not in {"berry", "solve_berry"}:
            raise ValueError("BCD source must be an explicit Model2D Berry record")
        summary_json = json.loads((berry_path / "summary.json").read_text(encoding="utf-8"))
        if summary_json.get("qualification", {}).get("overall_status") != "QUALIFIED":
            raise ValueError("BCD requires Berry qualification overall_status=QUALIFIED")
        with np.load(berry_path / "arrays.npz", allow_pickle=False) as arrays:
            plaquettes = np.asarray(arrays["plaquettes"])
            centers = np.mean(plaquettes, axis=1)
            curvature = np.asarray(arrays["curvature"])
        result = compute_berry_dipole(centers, curvature, frequency_window=parameters.frequency_window, frequency_samples=parameters.frequency_samples, response_weight=parameters.response_weights, occupation=parameters.occupation, record_root=root)
        summary = {"kind": result["kind"], "physical_response": result["physical_response"]}
        operation = "compute_berry_dipole"
    else:
        raise TypeError(f"unsupported direct parameter type: {type(parameters).__name__}")
    record = _single_new_record(root, before)
    style = as_dict(plot or PlotParameters())
    figure = plot_record(record, style)
    figure_path = export_figure(figure, record / "figures" / "default.png", width_px=style["width_px"], height_px=style["height_px"], dpi=style["dpi"])
    if show and not testing:
        show_figure(figure)
    return {"case": case_name, "operation": operation, "record_path": str(record), "figure_path": str(figure_path), "summary": summary}


def run_entrypoint(case_name: str, case: CaseParameters, parameters: Any) -> int:
    try:
        result = run_operation(case_name, case, parameters)
    except ValueError as exc:
        if isinstance(parameters, BCDParameters):
            print(json.dumps({"status": "blocked", "operation": "compute_berry_dipole", "message": str(exc)}))
            return 0
        raise
    print(json.dumps(result, default=str, sort_keys=True))
    return 0
