from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np

from legumephc.config import load_benchmark
from legumephc.diagnostics import relative_orbit_residual
from legumephc.geometry import geometry_spec, m7_orbit
from legumephc.records import create_run
from legumephc.solver import homogeneous_shell_frequencies, solve_homogeneous_pwe, solve_pwe


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows-native Legume C3 cross-check")
    parser.add_argument("--smoke", action="store_true", help="run the real three-point PWE pilot")
    parser.add_argument("--case", choices=("G16", "G15", "Circle"), default="G15")
    parser.add_argument("--gmax", type=float, default=2.0)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    config = load_benchmark()
    qpoints = m7_orbit(config)
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


if __name__ == "__main__":
    raise SystemExit(main())
