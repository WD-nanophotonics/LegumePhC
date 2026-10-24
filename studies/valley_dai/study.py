"""Dai FR0 valley Chern numbers over the truncated K/K' half Brillouin zones.

Runs a gmax ladder and a Wilson-stencil refinement on the legacy K' domain,
plus a K/K' time-reversal check, writing one immutable record per solve
under ``studies/valley_dai/results`` and a JSON summary to stdout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legumephc import solve_valley_chern
from legumephc.geometry import GeometrySpec, Lattice2D
from legumephc.model import Model2D
from studies.common import load_study_config, report, result_root


def dai_model(settings: dict) -> Model2D:
    geometry = settings["geometry"]
    lattice = Lattice2D.triangular(1.0)
    spec = GeometrySpec(
        name="DaiFR0", kind="polygon", radii=(float(geometry["triangle_circumradius_over_a"]),), sides=(3,),
        angles_degrees=(float(geometry["triangle_angle_degrees"]),), strict_c3=True,
        epsilon_background=float(geometry["epsilon_background"]), epsilon_inclusion=float(geometry["epsilon_hole"]),
        motif_kinds=("polygon",), motif_epsilons=(float(geometry["epsilon_hole"]),), direct_basis=lattice.direct_basis,
        centers=np.zeros((1, 2)),
    )
    return Model2D(spec, lattice)


def _row(result: dict, seconds: float) -> dict:
    summary = result["summary"]
    row = {"valley": summary["valley"], "gmax": summary["cutoff"]["gmax"], "plane_waves": summary["cutoff"]["closed_gvec_count"],
           "stencil": summary["stencil"], "seconds": round(seconds, 1), "record": result["record_path"].name, "bands": []}
    for band in summary["bands"]:
        row["bands"].append({
            "band_zero_based": band["band_zero_based"], "min_gap": band["min_gap"], "status": band["status"],
            "source_grid": band["SOURCE_GRID_MIDPOINT"]["valley_chern"], "clipped_cell": band["CLIPPED_CELL"]["valley_chern"],
            "qualified": f"{band['SOURCE_GRID_MIDPOINT']['qualified_samples']}/{band['SOURCE_GRID_MIDPOINT']['total_samples']}",
            "failure_reasons": band["SOURCE_GRID_MIDPOINT"]["failure_reasons"],
        })
    return row


def run() -> int:
    settings = load_study_config("valley_dai")
    model = dai_model(settings)
    records = result_root("valley_dai")
    domain = settings["domain"]
    common = {"bands": tuple(settings["bands_zero_based"]), "delta_k": domain["delta_k"], "delta_gamma": domain["delta_gamma"],
              "grid_denominator": domain["grid_denominator"], "gap_floor": settings["gap_floor"], "pol": settings["polarization"],
              "record_root": records}

    def solve(**options) -> dict:
        start = time.monotonic()
        result = solve_valley_chern(model, **common, **options)
        row = _row(result, time.monotonic() - start)
        print(f"  {row['valley']} gmax={row['gmax']} stencil={row['stencil']:.5f} N_G={row['plane_waves']} ({row['seconds']} s)", file=sys.stderr, flush=True)
        return row

    summary = {"model": model.geometry.name, "point_group": model.point_group, "legacy": settings["legacy_e9f_k_prime"], "paper": settings["paper_anchors_paper_k"]}
    summary["gmax_ladder"] = [solve(gmax=g, valley="K'") for g in settings["gmax_ladder"]]
    refine = settings["stencil_refinement"]
    summary["stencil_refinement"] = [solve(gmax=refine["gmax"], valley="K'", stencil=h) for h in refine["stencils"]]
    summary["time_reversal_partner"] = solve(gmax=settings["time_reversal_check_gmax"], valley="K")
    return report("valley_dai", summary)


if __name__ == "__main__":
    raise SystemExit(run())
