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
    c3_closed_reciprocal_basis,
    operator_covariance_residual,
    matrix_covariance_residual,
    rank1_wilson,
    rankn_wilson,
    reciprocal_c3_map,
    composite_scalar_density,
    reciprocal_basis_projector_residual,
    relative_orbit_residual,
    rotated_density_residual,
    scalar_field_density,
    te_scalar_field_densities,
)
from legumephc.geometry import DIRECT_BASIS, c3_geometry_residual, geometry_spec, m7_orbit, rotation
from legumephc.records import create_run
from legumephc.solver import (
    homogeneous_shell_frequencies,
    solve_homogeneous_pwe,
    solve_pwe,
    solve_pwe_closed_basis,
    solve_pwe_custom_basis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows-native Legume C3 cross-check")
    parser.add_argument("--smoke", action="store_true", help="run the real three-point PWE pilot")
    parser.add_argument("--case", choices=("G16", "G15", "Circle"), default="G15")
    parser.add_argument("--gmax", type=float, default=2.0)
    parser.add_argument("--convergence", action="store_true", help="scan all frozen geometries and gmax values")
    parser.add_argument("--extension", action="store_true", help="extend strict controls through gmax=6")
    parser.add_argument("--operator-diagnosis", action="store_true", help="diagnose exact C3 closure of the PWE reciprocal basis")
    parser.add_argument("--gmax7", action="store_true", help="run only the strict-control gmax=7 extension")
    parser.add_argument("--closed-adapter", action="store_true", help="test a C3-closed reciprocal-basis adapter")
    parser.add_argument("--closed-plaquette", action="store_true", help="run local Wilson and E/H scalars on a C3-closed basis")
    parser.add_argument("--bandpath", action="store_true", help="compare G15 Gamma-K-M-Gamma default and closed-basis paths")
    parser.add_argument("--berry-map", action="store_true", help="sample an independent G15 rank-2 Berry map")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    config = load_benchmark()
    qpoints = m7_orbit(config)
    if args.operator_diagnosis:
        return run_operator_diagnosis(config, args.results)
    if args.closed_adapter:
        return run_closed_adapter(config, args.results)
    if args.closed_plaquette:
        return run_closed_plaquette(config, args.results)
    if args.bandpath:
        return run_bandpath(config, args.results)
    if args.berry_map:
        return run_berry_map(config, args.results)
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
    fig.savefig(target / "figures" / "frequencies.png")
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
            fig.savefig(target / "figures" / "frequencies.png")
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


def run_closed_adapter(config, results_root: Path) -> int:
    """Test a symmetry-closed finite basis without modifying site-packages."""

    orbit = m7_orbit(config)
    transform = rotation(120.0)
    cases: dict[str, object] = {}
    for case in ("G15", "Circle"):
        spec = geometry_spec(config, case)
        standard = solve_pwe(spec, orbit, gmax=5.0, numeig=4, pol=config.raw["polarization"])
        closed_gvec = c3_closed_reciprocal_basis(standard["gvec"], orbit, rotation_matrix=transform)
        adapted = solve_pwe_custom_basis(spec, orbit, closed_gvec, numeig=4, pol=config.raw["polarization"])
        maps = []
        operator_residuals = []
        epsilon_residuals = []
        projector_residuals = []
        for index in range(3):
            next_index = (index + 1) % 3
            mapping = reciprocal_c3_map(
                adapted["gvec"], adapted["gvec"], orbit[index], orbit[next_index],
                rotation_matrix=transform, tolerance=1e-5,
            )
            maps.append({
                "matched_fraction": mapping["matched_fraction"],
                "maximum_matching_residual": mapping["maximum_matching_residual"],
                "unmatched_vectors": mapping["unmatched_vectors"],
            })
            operator_residuals.append(operator_covariance_residual(
                adapted["eps_inv_mat"], adapted["gvec"], orbit[index], orbit[next_index], mapping,
            ))
            epsilon_residuals.append(matrix_covariance_residual(adapted["eps_matrix"], mapping))
            projector_residuals.append(reciprocal_basis_projector_residual(
                adapted["eigenvectors"][index, :, 1:3], adapted["eigenvectors"][next_index, :, 1:3],
                adapted["gvec"], adapted["gvec"], orbit[index], orbit[next_index], rotation_matrix=transform,
            ))
        cases[case] = {
            "seed_gvec_count": int(standard["gvec"].shape[1]),
            "closed_gvec_count": int(closed_gvec.shape[1]),
            "frequency_c3_max_residual": max(relative_orbit_residual(adapted["frequencies"][:, band]) for band in range(4)),
            "projector_rank2_max_residual": max(projector_residuals),
            "operator_covariance_max_residual": max(operator_residuals),
            "epsilon_matrix_covariance_max_residual": max(epsilon_residuals),
            "transitions": maps,
        }
    report = {
        "status": "succeeded",
        "mode": "pwe_c3_closed_basis_adapter",
        "gmax_seed": 5.0,
        "cases": cases,
        "interpretation": "The adapter reconstructs the full Fourier epsilon matrix for an affine-C3-closed finite basis; Legume site-packages are unchanged.",
    }
    target = create_run(results_root, "pwe_c3_closed_basis_adapter", config.raw, report, {"qpoints_orbit": orbit})
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


def run_closed_plaquette(config, results_root: Path) -> int:
    """Run local Wilson and E/H scalar diagnostics on the closed basis."""

    orbit = m7_orbit(config)
    transform = rotation(120.0)
    steps = [float(value) for value in config.raw["scan"]["berry_steps"]]
    cases: dict[str, object] = {}
    for case in ("G15", "Circle", "G16"):
        spec = geometry_spec(config, case)
        seed = solve_pwe(spec, orbit, gmax=5.0, numeig=4, pol=config.raw["polarization"])
        closed_gvec = c3_closed_reciprocal_basis(seed["gvec"], orbit, rotation_matrix=transform)
        center_result = solve_pwe_custom_basis(spec, orbit, closed_gvec, numeig=4, pol=config.raw["polarization"])
        scalar_fields = te_scalar_field_densities(
            center_result["eigenvectors"], center_result["frequencies"], center_result["gvec"],
            center_result["eps_inv_mat"], orbit, basis=DIRECT_BASIS, grid_size=32,
        )
        scalar_residuals = {
            key: np.max(rotated_density_residual(value, orbit, basis=DIRECT_BASIS, rotation_matrix=transform), axis=0).tolist()
            for key, value in scalar_fields.items()
        }
        scalar_residuals["composite_23"] = float(np.max(rotated_density_residual(
            (scalar_fields["H"][..., 1:3].sum(axis=-1))[..., None], orbit,
            basis=DIRECT_BASIS, rotation_matrix=transform,
        )))
        step_summaries: dict[str, object] = {}
        for step in steps:
            plaquette, areas = _local_plaquette_points(orbit, step)
            result = solve_pwe_custom_basis(spec, plaquette, closed_gvec, numeig=4, pol=config.raw["polarization"])
            members = []
            for member in range(3):
                start = member * 4
                local_frequencies = result["frequencies"][start:start + 4]
                wilson = _wilson_summary(result, 4, target_band=1, offset=start)
                numerical = qualification(
                    local_frequencies,
                    [wilson["rank1"]["min_link_magnitude"]],
                    [wilson["rank2"]["min_singular_value"]],
                    [wilson["rank1"]["branch_margin"]],
                    [wilson["rank2"]["branch_margin"]],
                )
                members.append({
                    "member": member,
                    "signed_area": areas[member],
                    "phase_density_rank1": wilson["rank1"]["phase"] / areas[member],
                    "phase_density_rank2": wilson["rank2"]["phase"] / areas[member],
                    "wilson": wilson,
                    "association": {
                        "rank1": {"min_link_magnitude": wilson["rank1"]["min_link_magnitude"], "qualified": wilson["rank1"]["min_link_magnitude"] > 0.1},
                        "rank2": {"min_singular_value": wilson["rank2"]["min_singular_value"], "qualified": wilson["rank2"]["min_singular_value"] > 0.1},
                    },
                    "numerical_qualification": numerical,
                })
            step_summaries[str(step)] = {
                "plaquette_point_count": 12,
                "member_summaries": members,
                "pairwise_c3_residual": {
                    "rank1_phase_density_relative_residual": _pairwise_relative_residual([x["phase_density_rank1"] for x in members]),
                    "rank2_phase_density_relative_residual": _pairwise_relative_residual([x["phase_density_rank2"] for x in members]),
                    "rank1_link_relative_residual": _pairwise_relative_residual([x["wilson"]["rank1"]["min_link_magnitude"] for x in members]),
                    "rank2_link_relative_residual": _pairwise_relative_residual([x["wilson"]["rank2"]["min_singular_value"] for x in members]),
                },
            }
        cases[case] = {
            "seed_gvec_count": int(seed["gvec"].shape[1]),
            "closed_gvec_count": int(closed_gvec.shape[1]),
            "scalar_c3_residual_grid32": scalar_residuals,
            "berry_steps": step_summaries,
        }
    report = {
        "status": "succeeded",
        "mode": "pwe_c3_closed_local_plaquette",
        "gmax_seed": 5.0,
        "cases": cases,
        "interpretation": "Closed-basis local plaquettes use normalized PWE eigenvectors; rank-1 is withheld whenever gap, association/link, or branch qualification fails.",
    }
    target = create_run(results_root, "pwe_c3_closed_local_plaquette", config.raw, report, {"qpoints_orbit": orbit})
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


def _band_path(samples_per_segment: int = 5) -> np.ndarray:
    gamma = np.array([0.0, 0.0])
    k_point = np.array([2.0 / 3.0, 0.0])
    m_point = np.array([0.5, 1.0 / (2.0 * np.sqrt(3.0))])
    vertices = (gamma, k_point, m_point, gamma)
    points: list[np.ndarray] = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        points.extend(np.linspace(start, end, samples_per_segment, endpoint=False))
    points.append(vertices[-1])
    return np.asarray(points)


def run_bandpath(config, results_root: Path) -> int:
    """Compare the minimum Gamma-K-M-Gamma path in default and closed bases."""

    orbit = m7_orbit(config)
    path = _band_path(samples_per_segment=5)
    qbatch = np.vstack((path, orbit))
    rows: list[dict[str, object]] = []
    previous_default = previous_closed = None
    for gmax in (4.0, 5.0, 6.0):
        spec = geometry_spec(config, "G15")
        default = solve_pwe(spec, qbatch, gmax=gmax, numeig=4, pol=config.raw["polarization"])
        closed = solve_pwe_closed_basis(
            spec, qbatch, seed_gmax=gmax, closure_qpoints=orbit,
            numeig=4, pol=config.raw["polarization"],
        )
        default_path = default["frequencies"][:len(path)]
        closed_path = closed["frequencies"][:len(path)]
        default_orbit = default["frequencies"][len(path):]
        closed_orbit = closed["frequencies"][len(path):]
        row = {
            "gmax": gmax,
            "path_qpoints": path.tolist(),
            "default_path_frequencies": default_path.tolist(),
            "closed_path_frequencies": closed_path.tolist(),
            "max_default_closed_path_difference": float(np.max(np.abs(default_path - closed_path))),
            "default_path_change_from_previous": None if previous_default is None else float(np.max(np.abs(default_path - previous_default))),
            "closed_path_change_from_previous": None if previous_closed is None else float(np.max(np.abs(closed_path - previous_closed))),
            "default_m7_c3_residual": max(relative_orbit_residual(default_orbit[:, band]) for band in range(4)),
            "closed_m7_c3_residual": max(relative_orbit_residual(closed_orbit[:, band]) for band in range(4)),
            "closed_basis_counts": {"seed": closed["seed_gvec_count"], "closed": closed["closed_gvec_count"]},
        }
        rows.append(row)
        previous_default, previous_closed = default_path, closed_path
    report = {
        "status": "succeeded",
        "mode": "pwe_g15_bandpath_default_vs_closed",
        "case": "G15",
        "samples_per_segment": 5,
        "path": "Gamma-K-M-Gamma",
        "rows": rows,
        "closed_pilot_gate": "closed M7 C3 residual < 1e-6 for all tested gmax",
    }
    target = create_run(results_root, "pwe_g15_bandpath_default_vs_closed", config.raw, report, {"path_qpoints": path})
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=120)
    for row in rows:
        axes[0].plot(np.arange(len(path)), np.asarray(row["default_path_frequencies"])[:, 1], marker="o", label=f"default g{row['gmax']}")
        axes[1].plot(np.arange(len(path)), np.asarray(row["closed_path_frequencies"])[:, 1], marker="o", label=f"closed g{row['gmax']}")
    axes[0].set_title("G15 band 2 default")
    axes[1].set_title("G15 band 2 closed basis")
    for axis in axes:
        axis.set_xlabel("Γ–K–M–Γ sample")
        axis.set_ylabel("frequency (c/a)")
        axis.legend()
    figure.tight_layout()
    figure.savefig(target / "figures" / "bandpath.png")
    plt.close(figure)
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


def _map_grid(center: np.ndarray, size: int = 5, span: float = 0.1) -> np.ndarray:
    offsets = np.linspace(-span, span, size)
    return np.asarray([center + np.array([x, y]) for y in offsets for x in offsets])


def _unique_points(points: list[np.ndarray]) -> np.ndarray:
    unique: dict[tuple[float, float], np.ndarray] = {}
    for point in points:
        unique[tuple(np.round(point, 12))] = point
    return np.asarray(list(unique.values()))


def _berry_map_for_result(result: dict, grid: np.ndarray, solved_points: np.ndarray, *, basis: np.ndarray) -> dict[str, object]:
    index = {tuple(np.round(point, 12)): i for i, point in enumerate(solved_points)}
    frequencies = result["frequencies"]
    vectors = result["eigenvectors"]
    size = int(np.sqrt(len(grid)))
    rank2_phase = []
    rank1_phase = []
    quality = []
    for iy in range(size - 1):
        for ix in range(size - 1):
            corners = [
                grid[iy * size + ix],
                grid[iy * size + ix + 1],
                grid[(iy + 1) * size + ix + 1],
                grid[(iy + 1) * size + ix],
            ]
            indices = [index[tuple(np.round(point, 12))] for point in corners]
            local_rank1 = rank1_wilson([vectors[i, :, 1] for i in indices])
            local_rank2 = rankn_wilson([vectors[i, :, 1:3] for i in indices])
            dx = corners[1] - corners[0]
            dy = corners[3] - corners[0]
            area = float(dx[0] * dy[1] - dx[1] * dy[0])
            numerical = qualification(
                frequencies[indices],
                [local_rank1["min_link_magnitude"]],
                [local_rank2["min_singular_value"]],
                [local_rank1["branch_margin"]],
                [local_rank2["branch_margin"]],
            )
            rank2_phase.append({"iy": iy, "ix": ix, "signed_area": area, "phase": local_rank2["phase"], "phase_density": local_rank2["phase"] / area})
            rank1_phase.append({"iy": iy, "ix": ix, "signed_area": area, "phase": local_rank1["phase"], "phase_density": local_rank1["phase"] / area})
            quality.append({"iy": iy, "ix": ix, "rank1": {"qualified": False, "status": "RANK1_WITHHELD", "reason": "band-2 rank-1 Berry is withheld by the M7 pilot gap gate"}, "rank2": numerical["rank2"], "min_link": local_rank1["min_link_magnitude"], "min_singular": local_rank2["min_singular_value"]})
    return {"rank2_raw_map": rank2_phase, "rank1_status": "RANK1_WITHHELD", "quality_uncertainty_mask": quality}


def run_berry_map(config, results_root: Path) -> int:
    """Sample a small independent G15 Berry map in both reciprocal bases."""

    center = np.asarray(config.raw["m7"]["k_center_reciprocal"], dtype=float)
    orbit = m7_orbit(config)
    grid = _map_grid(center, size=5, span=0.1)
    rotated: list[np.ndarray] = []
    for point in grid:
        for angle in (0.0, 120.0, 240.0):
            rotated.append(center + rotation(angle) @ (point - center))
    solved_points = _unique_points(rotated)
    spec = geometry_spec(config, "G15")
    default = solve_pwe(spec, solved_points, gmax=5.0, numeig=4, pol=config.raw["polarization"])
    closed = solve_pwe_closed_basis(spec, solved_points, seed_gmax=5.0, closure_qpoints=orbit, numeig=4, pol=config.raw["polarization"])
    out = {}
    for label, result in (("default_circular", default), ("closed_c3", closed)):
        residual_map = []
        point_index = {tuple(np.round(point, 12)): i for i, point in enumerate(solved_points)}
        for point in grid:
            source = point_index[tuple(np.round(point, 12))]
            for angle in (120.0, 240.0):
                target_point = center + rotation(angle) @ (point - center)
                target = point_index[tuple(np.round(target_point, 12))]
                mapping = reciprocal_c3_map(
                    result["gvec"], result["gvec"], point, target_point, rotation_matrix=rotation(angle),
                )
                residual_map.append({
                    "point": point.tolist(), "rotation_degrees": angle,
                    "projector_residual": reciprocal_basis_projector_residual(
                        result["eigenvectors"][source, :, 1:3], result["eigenvectors"][target, :, 1:3],
                        result["gvec"], result["gvec"], point, target_point, rotation_matrix=rotation(angle),
                    ),
                    "matched_fraction": mapping["matched_fraction"],
                })
        out[label] = {"solved_point_count": len(solved_points), "berry": _berry_map_for_result(result, grid, solved_points, basis=DIRECT_BASIS), "c3_residual_map": residual_map}
    report = {
        "status": "succeeded", "mode": "pwe_g15_independent_berry_map", "case": "G15",
        "grid_size": 5, "grid_span_reciprocal": 0.1,
        "sampling": "25 independent grid points plus independently solved 120/240 degree images; no copied sectors or averaging",
        "bases": out,
    }
    target = create_run(results_root, "pwe_g15_independent_berry_map", config.raw, report, {"grid_qpoints": grid, "solved_qpoints": solved_points})
    figure, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=120)
    for column, (label, title) in enumerate((("default_circular", "default"), ("closed_c3", "closed C3"))):
        phase = np.asarray([item["phase_density"] for item in out[label]["berry"]["rank2_raw_map"]]).reshape(4, 4)
        residual = np.asarray([item["projector_residual"] for item in out[label]["c3_residual_map"]]).reshape(25, 2).max(axis=1).reshape(5, 5)
        axes[0, column].imshow(phase, origin="lower", aspect="auto")
        axes[0, column].set_title(f"{title} raw rank-2 phase density")
        axes[1, column].imshow(residual, origin="lower", aspect="auto")
        axes[1, column].set_title(f"{title} C3 projector residual")
    for axis in axes.flat:
        axis.set_xlabel("grid x")
        axis.set_ylabel("grid y")
    figure.tight_layout()
    figure.savefig(target / "figures" / "berry_map.png")
    plt.close(figure)
    print(json.dumps({"result_directory": str(target), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
