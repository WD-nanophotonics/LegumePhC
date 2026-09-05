from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legumephc import Affine2D, compute_berry_dipole, compute_field_observables, frequency_at_k, solve_bands, solve_berry, solve_efs, square_circle_spec
from legumephc.geometry import Lattice2D
from legumephc.model import Model2D
from studies.common import band_settings, load_study_config, report, result_root, synthetic_dipole_inputs

# Python band indices are zero-based; the configured target is explicit.


def run() -> int:
    settings = load_study_config("affine")
    if settings["base_lattice"] != "square" or settings["geometry"] != "SquareCircle":
        raise ValueError("affine study config must select the SquareCircle square base model")
    records = result_root("affine")
    affine = Affine2D(np.asarray(settings["linear"], dtype=float), np.asarray(settings["translation"], dtype=float))
    model = Model2D(square_circle_spec(radius=settings["radius"]), Lattice2D.square(), affine=affine)
    berry_bands, numeig = band_settings(settings)
    qpoints = np.asarray([settings["qpoint"]], dtype=float)
    summary = {"symmetry": model.point_group or "none", "polarizations": {}}
    for pol in settings["polarizations"]:
        bands = solve_bands(model, qpoints, gmax=settings["gmax"], numeig=numeig, pol=pol, record_root=records)
        path = solve_bands(model, path=settings["band_path"], gmax=settings["gmax"], numeig=numeig, pol=pol, record_root=records)
        efs = solve_efs(model, gmax=settings["gmax"], grid_size=settings["efs_grid_size"], bands=berry_bands, numeig=numeig, pol=pol, record_root=records)
        fields = compute_field_observables(bands, model, bands=(settings["target_band_zero_based"],), grid_size=settings["field_grid_size"], record_root=records)
        frequency = frequency_at_k(model, qpoints[0], gmax=settings["gmax"], band=settings["target_band_zero_based"], pol=pol, record_root=records)
        summary["polarizations"][pol] = {"frequency_at_k_band_zero_based": frequency, "band_path_labels": path["path_labels"], "efs_sampling": efs["sampling_domain"], "efs_grid_shape": efs["grid_shape"], "efs_inside_count": int(np.count_nonzero(efs["inside_bz_mask"])), "field_shape": list(fields["energy_density"].shape)}
    center = qpoints[0]
    step = settings["berry_step"]
    plaquette = center + np.asarray([[-step, -step], [-step, step], [step, step], [step, -step]])
    berry = solve_berry(model, plaquette, gmax=settings["gmax"], bands=berry_bands, rank=settings["berry_rank"], numeig=numeig, pol="te", record_root=records)
    synthetic_q, synthetic_curvature_base = synthetic_dipole_inputs(settings)
    dipole = compute_berry_dipole(synthetic_q, synthetic_curvature_base, record_root=records)
    summary.update({"berry_gate_status": berry["qualification"]["gate_status"], "berry_convergence_status": berry["qualification"]["convergence_status"], "berry_overall_status": berry["qualification"]["overall_status"], "berry_curvature": berry["curvature"].tolist(), "dipole_semantics": "SYNTHETIC_DEMONSTRATION", "dipole_kind": dipole["kind"], "effective_direct_basis": model.effective_lattice.direct_basis.tolist()})
    return report("affine", summary)


if __name__ == "__main__":
    raise SystemExit(run())
