from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legumephc.config import load_benchmark
from legumephc.diagnostics import (
    qualification,
    rank1_wilson,
    rankn_wilson,
    reciprocal_basis_projector_residual,
    relative_orbit_residual,
    rotated_density_residual,
    scalar_field_density,
)
from legumephc.geometry import DIRECT_BASIS, c3_geometry_residual, geometry_spec, m7_orbit, rotation
from legumephc.records import create_run
from legumephc.solver import homogeneous_shell_frequencies, solve_homogeneous_pwe, solve_pwe


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows-native Legume C3 cross-check")
    parser.add_argument("--smoke", action="store_true", help="run the real three-point PWE pilot")
    parser.add_argument("--case", choices=("G16", "G15", "Circle"), default="G15")
    parser.add_argument("--gmax", type=float, default=2.0)
    parser.add_argument("--convergence", action="store_true", help="scan all frozen geometries and gmax values")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    config = load_benchmark()
    qpoints = m7_orbit(config)
    if args.convergence:
        return run_pilot(config, args.results)
    if args.smoke:
        spec = geometry_spec(config, args.case)
        result = solve_pwe(spec, qpoints, gmax=args.gmax, numeig=4, pol=config.raw["polarization"])
        frequencies = result["frequencies"]
        residuals = [relative_orbit_residual(frequencies[:, band]) for band in range(frequencies.shape[1])]
        summary = {
            "status": "succeeded",
            "mode": "pwe_three_point_smoke",
            "case": args.case,
            "gmax": args.gmax,
            "frequency_c3_relative_residual_by_band": residuals,
            "frequencies": frequencies.tolist(),
            "python": platform.python_version(),
            "legume": result["legume_version"],
        }
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    else:
        shells = np.vstack([homogeneous_shell_frequencies(point, config.background_epsilon)[:4] for point in qpoints])
        result = solve_homogeneous_pwe(
            qpoints, config.background_epsilon, gmax=args.gmax, numeig=4,
            pol=config.raw["polarization"],
        )
        frequencies = result["frequencies"]
        absolute_error = np.abs(frequencies - shells)
        summary = {
            "status": "succeeded",
            "mode": "homogeneous_pwe_analytic_calibration",
            "gmax": args.gmax,
            "frequency_c3_relative_residual_by_band": [relative_orbit_residual(frequencies[:, i]) for i in range(4)],
            "maximum_absolute_analytic_error": float(np.max(absolute_error)),
            "pwe_frequencies": frequencies.tolist(),
            "analytic_frequencies": shells.tolist(),
            "k_coordinate_convention": "q Cartesian reciprocal without 2*pi; solver k = 2*pi*q",
            "python": platform.python_version(),
            "legume": result["legume_version"],
        }
        arrays = {"qpoints": qpoints, "pwe_frequencies": frequencies, "analytic_frequencies": shells, "gvec": result["gvec"]}

    target = create_run(args.results, summary["mode"], config.raw, summary, arrays)
    values = np.asarray(summary.get("frequencies", summary.get("pwe_frequencies")))
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    for band in range(values.shape[1]):
        ax.plot(range(3), values[:, band], marker="o", label=f"band {band + 1}")
    ax.set(xlabel="C3 orbit member", ylabel="frequency (c/a)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target / "frequencies.png")
    plt.close(fig)
    print(json.dumps({"result_directory": str(target), **summary}, indent=2))
    return 0


def _loop_points(qpoints: np.ndarray, step: float) -> np.ndarray:
    """Build a closed, piecewise-linear Berry loop with a declared step."""

    points: list[np.ndarray] = []
    for index in range(len(qpoints)):
        start, end = qpoints[index], qpoints[(index + 1) % len(qpoints)]
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
        points.extend(start + (end - start) * (n / count) for n in range(count))
    return np.asarray(points)


def _wilson_summary(result: dict, loop_count: int, target_band: int = 1, offset: int = 0) -> dict[str, object]:
    vectors = result["eigenvectors"]
    indices = range(offset, offset + loop_count)
    rank1 = rank1_wilson([vectors[index, :, target_band] for index in indices])
    rank2 = rankn_wilson([vectors[index, :, target_band:target_band + 2] for index in indices])
    return {
        "rank1": rank1,
        "rank2": rank2,
        "rank1_link_magnitude": rank1["min_link_magnitude"],
        "rank2_link_singular_value": rank2["min_singular_value"],
    }


def run_pilot(config, results_root: Path) -> int:
    """Run the bounded three-geometry C3 convergence and Wilson pilot."""

    center = np.asarray(config.raw["m7"]["k_center_reciprocal"], dtype=float)
    orbit = m7_orbit(config)
    all_summaries: dict[str, object] = {}
    steps = [float(value) for value in config.raw["scan"]["berry_steps"]]
    gmax_values = [float(value) for value in config.raw["scan"]["gmax"]]
    for case in ("G15", "Circle", "G16"):
        spec = geometry_spec(config, case)
        case_rows: list[dict[str, object]] = []
        previous: np.ndarray | None = None
        for gmax in gmax_values:
            # The center and all three orbit points are intentionally solved
            # in one Legume batch for this geometry/cutoff.
            qbatch = np.vstack((center, orbit))
            result = solve_pwe(spec, qbatch, gmax=gmax, numeig=4, pol=config.raw["polarization"])
            frequencies = result["frequencies"]
            orbit_frequencies = frequencies[1:]
            densities = scalar_field_density(
                result["eigenvectors"][1:], result["gvec"], orbit,
                basis=DIRECT_BASIS, grid_size=32,
            )
            density_residuals = rotated_density_residual(
                densities, orbit, basis=DIRECT_BASIS, rotation_matrix=rotation(120.0),
            )
            projector_residuals = []
            for index in range(3):
                source = result["eigenvectors"][1 + index, :, 1:3]
                target = result["eigenvectors"][1 + ((index + 1) % 3), :, 1:3]
                projector_residuals.append(reciprocal_basis_projector_residual(
                    source, target, result["gvec"], result["gvec"],
                    orbit[index], orbit[(index + 1) % 3], rotation_matrix=rotation(120.0),
                ))
            direct_wilson = _wilson_summary(result, loop_count=3, target_band=1, offset=1)
            max_frequency_residual = max(
                relative_orbit_residual(orbit_frequencies[:, band]) for band in range(4)
            )
            frequency_change = None if previous is None else float(np.max(np.abs(orbit_frequencies - previous)))
            row: dict[str, object] = {
                "gmax": gmax,
                "frequencies": orbit_frequencies.tolist(),
                "frequency_c3_relative_residual_by_band": [
                    relative_orbit_residual(orbit_frequencies[:, band]) for band in range(4)
                ],
                "frequency_change_from_previous_gmax": frequency_change,
                "frequency_converged": bool(
                    frequency_change is not None
                    and frequency_change <= 2e-3
                    and max_frequency_residual <= 1e-3
                ),
                "density_rotated_residual_by_band": np.max(density_residuals, axis=0).tolist(),
                "projector_rank2_covariance_residual_by_link": projector_residuals,
                "direct_three_point_wilson": direct_wilson,
                "geometry_residual": float(c3_geometry_residual(spec)),
                "berry_steps": {},
            }
            previous = orbit_frequencies.copy()
            for step in steps:
                loop = _loop_points(orbit, step)
                # The center plus the complete discretized loop are one batch.
                loop_result = solve_pwe(spec, np.vstack((center, loop)), gmax=gmax, numeig=4, pol=config.raw["polarization"])
                loop_summary = _wilson_summary(loop_result, loop_count=len(loop), target_band=1, offset=1)
                loop_summary["point_count"] = len(loop)
                loop_summary["qualification"] = qualification(
                    loop_result["frequencies"],
                    [loop_summary["rank1"]["min_link_magnitude"]],
                    [loop_summary["rank2"]["min_singular_value"]],
                    [loop_summary["rank1"]["branch_margin"]],
                    [loop_summary["rank2"]["branch_margin"]],
                )
                row["berry_steps"][str(step)] = loop_summary
            row["qualification"] = qualification(
                frequencies,
                [direct_wilson["rank1"]["min_link_magnitude"]],
                [direct_wilson["rank2"]["min_singular_value"]],
                [direct_wilson["rank1"]["branch_margin"]],
                [direct_wilson["rank2"]["branch_margin"]],
            )
            case_rows.append(row)
            arrays = {
                "qpoints_center_orbit": qbatch,
                "frequencies": frequencies,
                "density_orbit_band2": densities[:, :, :, 1],
            }
            summary = {
                "status": "succeeded",
                "mode": "pwe_c3_pilot",
                "case": case,
                "gmax": gmax,
                "row": row,
                "python": platform.python_version(),
                "legume": result["legume_version"],
            }
            target = create_run(results_root, "pwe_c3_pilot", config.raw, summary, arrays)
            values = frequencies
            fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
            for band in range(values.shape[1]):
                ax.plot(range(values.shape[0]), values[:, band], marker="o", label=f"band {band + 1}")
            ax.set(xlabel="center + C3 orbit point", ylabel="frequency (c/a)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(target / "frequencies.png")
            plt.close(fig)
        all_summaries[case] = case_rows
    report = {"status": "succeeded", "mode": "pwe_c3_convergence", "cases": all_summaries, "python": platform.python_version()}
    target = create_run(results_root, "pwe_c3_convergence", config.raw, report, {"qpoints_orbit": orbit})
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
