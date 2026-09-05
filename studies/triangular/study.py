from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legumephc import compute_berry_dipole, compute_field_observables, frequency_at_k, solve_bands, solve_berry, solve_efs
from legumephc.config import load_benchmark
from legumephc.geometry import Lattice2D, m7_orbit
from legumephc.model import Model2D
from studies.common import band_settings, load_study_config, report, result_root, synthetic_dipole_inputs

# Python band indices are zero-based; the configured target is explicit.


def _synthetic_dipole(settings: dict, record_root: Path) -> dict:
    qpoints, curvature = synthetic_dipole_inputs(settings)
    return compute_berry_dipole(qpoints, curvature, q_units="reduced", record_root=record_root)


def run() -> int:
    settings = load_study_config("triangular")
    benchmark = load_benchmark((ROOT / "studies" / "triangular" / settings["benchmark_config"]).resolve())
    records = result_root("triangular")
    orbit = m7_orbit(benchmark)
    berry_bands, numeig = band_settings(settings)
    summary = {"lattice": "triangular", "cases": {}}
    for name in settings["cases"]:
        model = Model2D.from_benchmark(benchmark, name, lattice=Lattice2D.triangular())
        bands = solve_bands(model, orbit, gmax=settings["gmax"], numeig=numeig, pol=settings["polarization"], record_root=records)
        path = solve_bands(model, path=settings["band_path"], gmax=settings["gmax"], numeig=numeig, pol=settings["polarization"], record_root=records)
        efs = solve_efs(model, gmax=settings["gmax"], grid_size=settings["efs_grid_size"], bands=berry_bands, numeig=numeig, pol=settings["polarization"], record_root=records)
        fields = compute_field_observables(bands, model, bands=berry_bands, grid_size=settings["field_grid_size"], record_root=records)
        frequency = frequency_at_k(model, orbit[0], gmax=settings["gmax"], band=settings["target_band_zero_based"], pol=settings["polarization"], record_root=records)
        item = {"symmetry": model.point_group or "none", "frequency_at_k_band_zero_based": frequency, "band_path_labels": path["path_labels"], "efs_sampling": efs["sampling_domain"], "efs_grid_shape": efs["grid_shape"], "efs_inside_count": int(np.count_nonzero(efs["inside_bz_mask"])), "field_shape": list(fields["energy_density"].shape)}
        if name in settings["berry_cases"]:
            center = orbit[0]
            step = settings["berry_step"]
            plaquette = center + np.asarray([[-step, -step], [-step, step], [step, step], [step, -step]])
            berry = solve_berry(model, plaquette, gmax=settings["gmax"], bands=berry_bands, rank=settings["berry_rank"], numeig=numeig, pol=settings["polarization"], record_root=records)
            dipole = _synthetic_dipole(settings, records)
            item.update({"berry_gate_status": berry["qualification"]["gate_status"], "berry_convergence_status": berry["qualification"]["convergence_status"], "berry_overall_status": berry["qualification"]["overall_status"], "berry_curvature": berry["curvature"].tolist(), "dipole_semantics": "SYNTHETIC_DEMONSTRATION", "dipole_kind": dipole["kind"]})
        summary["cases"][name] = item
    return report("triangular", summary)


if __name__ == "__main__":
    raise SystemExit(run())
