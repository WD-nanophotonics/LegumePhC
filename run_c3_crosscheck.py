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
    operator_covariance_residual,
    rank1_wilson,
    rankn_wilson,
    reciprocal_c3_map,
    composite_scalar_density,
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
    parser.add_argument("--extension", action="store_true", help="extend strict controls through gmax=6")
    parser.add_argument("--operator-diagnosis", action="store_true", help="diagnose exact C3 closure of the PWE reciprocal basis")
    parser.add_argument("--gmax7", action="store_true", help="run only the strict-control gmax=7 extension")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    config = load_benchmark()
    qpoints = m7_orbit(config)
    if args.operator_diagnosis:
        return run_operator_diagnosis(config, args.results)
    if args.convergence:
        return run_pilot(config, args.results)
    if args.extension:
        return run_pilot(
            config, args.results, cases=("G15", "Circle"),
            gmax_values=(4.0, 5.0, 6.0), mode="pwe_c3_extension",
        )
    if args.gmax7:
        return run_pilot(
            config, args.results, cases=("G15", "Circle"),
            gmax_values=(7.0,), mode="pwe_c3_gmax7",
        )
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


def _local_plaquette_points(centers: np.ndarray, step: float) -> tuple[np.ndarray, list[float]]:
    """Return three C3-covariant four-corner local plaquettes.

    The side vectors at member zero are rotated together with its center. The
    returned order is counter-clockwise for each member and is suitable for a
    four-link Wilson loop.
    """

    base_dx = np.array([step, 0.0])
    base_dy = np.array([0.0, step])
    corners: list[np.ndarray] = []
    areas: list[float] = []
    for index, center in enumerate(centers):
        transform = rotation(120.0 * index)
        dx, dy = transform @ base_dx, transform @ base_dy
        corners.extend((center, center + dx, center + dx + dy, center + dy))
        areas.append(float(dx[0] * dy[1] - dx[1] * dy[0]))
    return np.asarray(corners), areas


def _pairwise_relative_residual(values: list[float]) -> float:
    data = np.asarray(values, dtype=float)
    scale = max(float(np.max(np.abs(data))), np.finfo(float).eps)
    return float(np.max(data) - np.min(data)) / scale


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


def run_pilot(
    config,
    results_root: Path,
    *,
    cases: tuple[str, ...] = ("G15", "Circle", "G16"),
    gmax_values: tuple[float, ...] | None = None,
    mode: str = "pwe_c3_convergence",
) -> int:
    """Run the bounded three-geometry C3 convergence and Wilson pilot."""

    center = np.asarray(config.raw["m7"]["k_center_reciprocal"], dtype=float)
    orbit = m7_orbit(config)
    all_summaries: dict[str, object] = {}
    steps = [float(value) for value in config.raw["scan"]["berry_steps"]]
    if gmax_values is None:
        gmax_values = tuple(float(value) for value in config.raw["scan"]["gmax"])
    for case in cases:
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
            densities64 = scalar_field_density(
                result["eigenvectors"][1:], result["gvec"], orbit,
                basis=DIRECT_BASIS, grid_size=64,
            )
            density_residuals32 = rotated_density_residual(
                densities, orbit, basis=DIRECT_BASIS, rotation_matrix=rotation(120.0),
            )
            density_residuals64 = rotated_density_residual(
                densities64, orbit, basis=DIRECT_BASIS, rotation_matrix=rotation(120.0),
            )
            composite32 = composite_scalar_density(
                result["eigenvectors"][1:], result["gvec"], orbit,
                bands=(1, 2), basis=DIRECT_BASIS, grid_size=32,
            )
            composite64 = composite_scalar_density(
                result["eigenvectors"][1:], result["gvec"], orbit,
                bands=(1, 2), basis=DIRECT_BASIS, grid_size=64,
            )
            composite_residual32 = rotated_density_residual(
                composite32[..., None], orbit, basis=DIRECT_BASIS, rotation_matrix=rotation(120.0),
            )
            composite_residual64 = rotated_density_residual(
                composite64[..., None], orbit, basis=DIRECT_BASIS, rotation_matrix=rotation(120.0),
            )
            projector_residuals = []
            for index in range(3):
                source = result["eigenvectors"][1 + index, :, 1:3]
                target = result["eigenvectors"][1 + ((index + 1) % 3), :, 1:3]
                projector_residuals.append(reciprocal_basis_projector_residual(
                    source, target, result["gvec"], result["gvec"],
                    orbit[index], orbit[(index + 1) % 3], rotation_matrix=rotation(120.0),
                ))
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
                "density_rotated_residual_by_band_grid32": np.max(density_residuals32, axis=0).tolist(),
                "density_rotated_residual_by_band_grid64": np.max(density_residuals64, axis=0).tolist(),
                "composite_23_density_rotated_residual_grid32": float(np.max(composite_residual32)),
                "composite_23_density_rotated_residual_grid64": float(np.max(composite_residual64)),
                "projector_rank2_covariance_residual_by_link": projector_residuals,
                "geometry_residual": float(c3_geometry_residual(spec)),
                "berry_steps": {},
            }
            previous = orbit_frequencies.copy()
            for step in steps:
                plaquette, areas = _local_plaquette_points(orbit, step)
                # The 12 corners (three local plaquettes) are solved in one
                # batch; no global orbit path is used for Berry qualification.
                plaquette_result = solve_pwe(spec, plaquette, gmax=gmax, numeig=4, pol=config.raw["polarization"])
                member_summaries = []
                for member in range(3):
                    start = member * 4
                    local_frequencies = plaquette_result["frequencies"][start:start + 4]
                    local_wilson = _wilson_summary(
                        plaquette_result, loop_count=4, target_band=1, offset=start,
                    )
                    numerical = qualification(
                        local_frequencies,
                        [local_wilson["rank1"]["min_link_magnitude"]],
                        [local_wilson["rank2"]["min_singular_value"]],
                        [local_wilson["rank1"]["branch_margin"]],
                        [local_wilson["rank2"]["branch_margin"]],
                    )
                    member_summaries.append({
                        "member": member,
                        "signed_area": areas[member],
                        "phase_density_rank1": local_wilson["rank1"]["phase"] / areas[member],
                        "phase_density_rank2": local_wilson["rank2"]["phase"] / areas[member],
                        "wilson": local_wilson,
                        "numerical_qualification": numerical,
                    })
                rank1_densities = [item["phase_density_rank1"] for item in member_summaries]
                rank2_densities = [item["phase_density_rank2"] for item in member_summaries]
                rank1_links = [item["wilson"]["rank1"]["min_link_magnitude"] for item in member_summaries]
                rank2_links = [item["wilson"]["rank2"]["min_singular_value"] for item in member_summaries]
                pairwise = {
                    "rank1_phase_density_relative_residual": _pairwise_relative_residual(rank1_densities),
                    "rank2_phase_density_relative_residual": _pairwise_relative_residual(rank2_densities),
                    "rank1_link_relative_residual": _pairwise_relative_residual(rank1_links),
                    "rank2_link_relative_residual": _pairwise_relative_residual(rank2_links),
                }
                row["berry_steps"][str(step)] = {
                    "plaquette_point_count": 12,
                    "member_summaries": member_summaries,
                    "pairwise_c3_residual": pairwise,
                }
            case_rows.append(row)
            arrays = {
                "qpoints_center_orbit": qbatch,
                "frequencies": frequencies,
                "density_orbit_band2_grid32": densities[:, :, :, 1],
                "density_orbit_composite_23_grid32": composite32,
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
    covariance_basis: dict[str, object] = {}
    for case, rows in all_summaries.items():
        by_gmax = {float(row["gmax"]): row for row in rows}
        if 4.0 in by_gmax and 5.0 in by_gmax:
            r4 = max(by_gmax[4.0]["projector_rank2_covariance_residual_by_link"])
            r5 = max(by_gmax[5.0]["projector_rank2_covariance_residual_by_link"])
            variation = abs(r5 - r4)
            covariance_basis[case] = {
                "gmax4_max_residual": r4,
                "gmax5_max_residual": r5,
                "observed_cutoff_variation": variation,
                "tolerance_basis": "absolute gmax=4 to gmax=5 variation; no fixed C3 tolerance",
                "gmax5_status": "C3_NOT_QUALIFIED" if r5 > variation else "C3_QUALIFIED",
            }
            for row in rows:
                if float(row["gmax"]) >= 5.0:
                    row["c3_covariance_qualification"] = {
                        "projector_max_residual": max(row["projector_rank2_covariance_residual_by_link"]),
                        "tolerance": variation,
                        "status": covariance_basis[case]["gmax5_status"],
                    }
    report = {
        "status": "succeeded", "mode": mode, "cases": all_summaries,
        "projector_c3_tolerance_basis": covariance_basis,
        "python": platform.python_version(),
    }
    target = create_run(results_root, mode, config.raw, report, {"qpoints_orbit": orbit})
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


def run_operator_diagnosis(config, results_root: Path) -> int:
    """Check reciprocal-basis closure and TE operator covariance at gmax 2..6."""

    orbit = m7_orbit(config)
    rows_by_case: dict[str, list[dict[str, object]]] = {}
    transform = rotation(120.0)
    for case in ("G15", "Circle"):
        spec = geometry_spec(config, case)
        rows: list[dict[str, object]] = []
        for gmax in (2.0, 3.0, 4.0, 5.0, 6.0):
            result = solve_pwe(spec, orbit, gmax=gmax, numeig=4, pol=config.raw["polarization"])
            transitions: list[dict[str, object]] = []
            projector_residuals: list[float] = []
            operator_residuals: list[float] = []
            for index in range(3):
                next_index = (index + 1) % 3
                mapping = reciprocal_c3_map(
                    result["gvec"], result["gvec"], orbit[index], orbit[next_index],
                    rotation_matrix=transform,
                )
                operator_residual = operator_covariance_residual(
                    result["eps_inv_mat"], result["gvec"], orbit[index], orbit[next_index], mapping,
                )
                if operator_residual is not None:
                    operator_residuals.append(operator_residual)
                projector_residuals.append(reciprocal_basis_projector_residual(
                    result["eigenvectors"][index, :, 1:3],
                    result["eigenvectors"][next_index, :, 1:3],
                    result["gvec"], result["gvec"], orbit[index], orbit[next_index],
                    rotation_matrix=transform,
                ))
                transitions.append({
                    "source_member": index,
                    "target_member": next_index,
                    "matched_count": mapping["matched_count"],
                    "source_count": mapping["source_count"],
                    "matched_fraction": mapping["matched_fraction"],
                    "unmatched_vectors": mapping["unmatched_vectors"],
                    "maximum_matching_residual": mapping["maximum_matching_residual"],
                    "shift_reciprocal": mapping["shift_reciprocal"],
                    "operator_covariance_relative_residual": operator_residual,
                })
            rows.append({
                "gmax": gmax,
                "frequency_c3_max_residual": max(relative_orbit_residual(result["frequencies"][:, band]) for band in range(4)),
                "projector_rank2_max_residual": max(projector_residuals),
                "minimum_matched_fraction": min(item["matched_fraction"] for item in transitions),
                "maximum_unmatched_count": max(item["source_count"] - item["matched_count"] for item in transitions),
                "operator_covariance_max_residual": max(operator_residuals) if operator_residuals else None,
                "transitions": transitions,
            })
        rows_by_case[case] = rows
    report = {
        "status": "succeeded",
        "mode": "pwe_reciprocal_c3_operator_diagnosis",
        "cases": rows_by_case,
        "interpretation": "C3 map includes q_target - R q_source = R*K-K reciprocal shift; missing basis vectors are truncation-boundary evidence.",
    }
    target = create_run(results_root, "pwe_reciprocal_c3_operator_diagnosis", config.raw, report, {"qpoints_orbit": orbit})
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
