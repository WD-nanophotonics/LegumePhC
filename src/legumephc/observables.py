from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import GeometrySpec, Lattice2D, polygon_vertices
from .records import create_model_record, create_record


def _grid(basis: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    fractional = np.stack(np.meshgrid(
        np.arange(size, dtype=float) / size,
        np.arange(size, dtype=float) / size,
        indexing="xy",
    ), axis=-1)
    return fractional, fractional @ np.asarray(basis, dtype=float).T


def _motif_mask(points: np.ndarray, spec: GeometrySpec, basis: np.ndarray, index: int) -> np.ndarray:
    center = np.asarray(spec.centers[index], dtype=float)
    mask = np.zeros(len(points), dtype=bool)
    for n1 in range(-1, 2):
        for n2 in range(-1, 2):
            shift = np.asarray(basis, dtype=float) @ np.array([n1, n2], dtype=float)
            local_points = points - center - shift
            if spec.kind == "circle":
                ellipse = spec.ellipse_parameters[index] if spec.ellipse_parameters else None
                if ellipse is None:
                    mask |= np.sum(local_points * local_points, axis=1) <= spec.radii[index] ** 2
                else:
                    rx, ry, phi = ellipse
                    x1 = local_points[:, 0] * np.cos(phi) + local_points[:, 1] * np.sin(phi)
                    y1 = -local_points[:, 0] * np.sin(phi) + local_points[:, 1] * np.cos(phi)
                    mask |= (x1 / rx) ** 2 + (y1 / ry) ** 2 <= 1.0
            else:
                assert spec.sides is not None
                if spec.transformed_vertices is None:
                    vertices = polygon_vertices(spec.radii[index], spec.sides[index], spec.angles_degrees[index], center)
                else:
                    vertices = spec.transformed_vertices[index]
                    local_points = points - shift
                x, y = local_points[:, 0], local_points[:, 1]
                inside = np.zeros(len(points), dtype=bool)
                for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                    crossing = ((start[1] > y) != (end[1] > y)) & (
                        x < (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1] + 1e-30) + start[0]
                    )
                    inside ^= crossing
                mask |= inside
    return mask


def _material_grid(spec: GeometrySpec, basis: np.ndarray, fractional: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = (fractional @ np.asarray(basis, dtype=float).T).reshape(-1, 2)
    regions = np.asarray([_motif_mask(points, spec, basis, index) for index in range(len(spec.radii))])
    union = np.any(regions, axis=0)
    epsilon = np.full(len(points), spec.epsilon_background, dtype=float)
    for index, region in enumerate(regions):
        epsilon[region] = spec.epsilon_inclusion
    return epsilon.reshape(fractional.shape[:-1]), np.vstack((regions, ~union)).reshape(len(regions) + 1, *fractional.shape[:-1])


def compute_field_observables(
    result: dict[str, Any],
    spec: GeometrySpec | Any,
    *,
    lattice: Lattice2D | None = None,
    bands: tuple[int, ...] | None = None,
    grid_size: int = 32,
    record_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute gauge-invariant field scalars and region energy ratios."""

    model = spec if hasattr(spec, "geometry") else None
    spec = model.geometry if model is not None else spec
    if lattice is None and model is not None:
        lattice = model.effective_lattice
    basis = spec.direct_basis if lattice is None else lattice.direct_basis
    fractional, cart = _grid(basis, grid_size)
    epsilon, regions = _material_grid(spec, basis, fractional)
    vectors = np.asarray(result["eigenvectors"], dtype=complex)
    frequencies = np.asarray(result["frequencies"], dtype=float)
    gvec = np.asarray(result["gvec"], dtype=float)
    kpoints = np.asarray(result.get("kpoints_cartesian"), dtype=float)
    polarization = str(result.get("polarization", "te")).lower()
    selected = tuple(range(vectors.shape[-1])) if bands is None else tuple(int(band) for band in bands)
    if any(band < 0 or band >= vectors.shape[-1] for band in selected):
        raise IndexError("requested band is outside the solved eigenstate array")
    kplusg = kpoints[:, None, :] + gvec.T[None, :, :]
    phases = np.exp(1j * np.einsum("kga,pa->kgp", kplusg, cart.reshape(-1, 2)))
    coefficients = vectors[:, :, selected]
    scalar = np.einsum("kgp,kgb->kpb", phases, coefficients).reshape(len(kpoints), grid_size, grid_size, len(selected))
    omega = 2.0 * np.pi * frequencies[:, selected]
    safe_omega = np.where(np.abs(omega) > 1e-14, omega, np.inf)
    if polarization == "te":
        perpendicular = np.stack((kplusg[..., 1], -kplusg[..., 0]), axis=-1)
        d_coeff = 1j * coefficients[..., None] * perpendicular[:, :, None, :] / safe_omega[:, None, :, None]
        d_x, d_y = d_coeff[..., 0], d_coeff[..., 1]
        eps_inv = np.asarray(result["eps_inv_mat"], dtype=complex)
        e_x = np.einsum("ij,kjb->kib", eps_inv, d_x)
        e_y = np.einsum("ij,kjb->kib", eps_inv, d_y)
        dxf = np.einsum("kgp,kgb->kpb", phases, d_x).reshape(len(kpoints), grid_size, grid_size, len(selected))
        dyf = np.einsum("kgp,kgb->kpb", phases, d_y).reshape(len(kpoints), grid_size, grid_size, len(selected))
        exf = np.einsum("kgp,kgb->kpb", phases, e_x).reshape(len(kpoints), grid_size, grid_size, len(selected))
        eyf = np.einsum("kgp,kgb->kpb", phases, e_y).reshape(len(kpoints), grid_size, grid_size, len(selected))
        h2 = np.abs(scalar) ** 2
        e2 = np.abs(exf) ** 2 + np.abs(eyf) ** 2
        energy = 0.5 * (np.real(np.conj(exf) * dxf + np.conj(eyf) * dyf) + h2)
    elif polarization == "tm":
        h_coeff = 1j * coefficients[..., None] * np.stack((kplusg[..., 1], -kplusg[..., 0]), axis=-1)[:, :, None, :] / safe_omega[:, None, :, None]
        h_x = np.einsum("kgp,kgb->kpb", phases, h_coeff[..., 0]).reshape(len(kpoints), grid_size, grid_size, len(selected))
        h_y = np.einsum("kgp,kgb->kpb", phases, h_coeff[..., 1]).reshape(len(kpoints), grid_size, grid_size, len(selected))
        e2 = np.abs(scalar) ** 2
        h2 = np.abs(h_x) ** 2 + np.abs(h_y) ** 2
        energy = 0.5 * (epsilon[None, :, :, None] * e2 + h2)
    else:
        raise ValueError("polarization must be TE or TM")
    for value in (e2, h2, energy):
        value /= np.maximum(np.mean(value, axis=(1, 2), keepdims=True), 1e-30)
    ratios = np.empty((len(kpoints), len(selected), regions.shape[0]), dtype=float)
    for region_index, region in enumerate(regions):
        ratios[:, :, region_index] = np.mean(energy * region[None, :, :, None], axis=(1, 2))
    ratios /= np.maximum(np.sum(ratios, axis=-1, keepdims=True), 1e-30)
    composite_energy = np.sum(energy, axis=-1)
    composite_ratios = np.asarray([
        np.mean(composite_energy * region[None, :, :], axis=(1, 2)) for region in regions
    ]).transpose(1, 0)
    composite_ratios /= np.maximum(np.sum(composite_ratios, axis=-1, keepdims=True), 1e-30)
    output = {
        "E2": e2,
        "H2": h2,
        "energy_density": energy,
        "region_energy_ratios": ratios,
        "composite_energy_density": composite_energy,
        "composite_region_energy_ratios": composite_ratios,
        "region_count": regions.shape[0],
        "bands": selected,
        "polarization": polarization,
        "gauge_invariant": True,
    }
    if record_root is not None:
        if model is not None:
            create_model_record(record_root, model, "compute_field_observables", {"grid_size": grid_size, "bands": selected, "polarization": polarization}, {"status": "succeeded", "gauge_invariant": True, "region_count": regions.shape[0]}, {key: value for key, value in output.items() if isinstance(value, np.ndarray)})
            return output
        basis_identity = np.asarray(spec.direct_basis if lattice is None else lattice.direct_basis)
        identity = {
            "model": "Model2D", "geometry": spec.name,
            "affine": {"linear": np.eye(2).tolist(), "translation": [0.0, 0.0]},
            "basis": {"policy": "solved", "direct_basis": basis_identity.tolist()},
            "solver": "Legume.PlaneWaveExp", "operation": "compute_field_observables",
        }
        create_record(record_root, identity=identity, config={"grid_size": grid_size, "bands": selected, "polarization": polarization}, summary={"status": "succeeded", "gauge_invariant": True, "region_count": regions.shape[0]}, arrays={key: value for key, value in output.items() if isinstance(value, np.ndarray)})
    return output
