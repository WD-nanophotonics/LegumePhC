from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import (
    DIRECT_BASIS,
    GeometrySpec,
    point_group_operations,
    polygon_vertices,
)

EIGENSOLVER_SHIFT = 1.0


def build_layer(spec: GeometrySpec, *, lattice=None):
    import legume

    direct_basis = spec.direct_basis if lattice is None else np.asarray(lattice.direct_basis, dtype=float)
    a1, a2 = direct_basis.T
    lattice = legume.Lattice(a1, a2)
    layer = legume.ShapesLayer(lattice, eps_b=spec.epsilon_background)
    for index, (center, radius) in enumerate(zip(spec.centers, spec.radii)):
        if spec.kind == "circle":
            shape = legume.Circle(eps=spec.epsilon_inclusion, x_cent=float(center[0]), y_cent=float(center[1]), r=radius)
        else:
            assert spec.sides is not None
            vertices = polygon_vertices(radius, spec.sides[index], spec.angles_degrees[index], center)
            shape = legume.Poly(eps=spec.epsilon_inclusion, x_edges=vertices[:, 0], y_edges=vertices[:, 1])
        layer.add_shape(shape)
    return lattice, layer


def q_to_legume_k(qpoints: np.ndarray) -> np.ndarray:
    """Convert MePhC's Cartesian reciprocal coordinates to Legume units.

    The frozen K=(2/3, 0) convention is already Cartesian in inverse lattice
    constants, with 2*pi omitted.  It is not a pair of reciprocal-basis
    coefficients.
    """

    return 2 * math.pi * np.asarray(qpoints, dtype=float)


def solve_pwe(spec: GeometrySpec, qpoints: np.ndarray, *, gmax: float, numeig: int = 4, pol: str = "te", lattice=None) -> dict[str, Any]:
    import legume

    _, layer = build_layer(spec, lattice=lattice)
    pwe = legume.PlaneWaveExp(layer, gmax=gmax)
    kpoints = q_to_legume_k(qpoints).T
    pwe.run(kpoints=kpoints, pol=pol.lower(), numeig=numeig)
    return {
        "frequencies": np.asarray(pwe.freqs),
        "eigenvectors": np.asarray(pwe.eigvecs),
        "gvec": np.asarray(pwe.gvec),
        "eps_inv_mat": np.asarray(pwe.eps_inv_mat),
        "kpoints_cartesian": kpoints.T,
        "polarization": pol.lower(),
        "eigensolver_shift": EIGENSOLVER_SHIFT,
        "legume_version": getattr(legume, "__version__", "unknown"),
    }


def solve_pwe_custom_basis(
    spec: GeometrySpec,
    qpoints: np.ndarray,
    gvec: np.ndarray,
    *,
    numeig: int = 4,
    pol: str = "te",
    lattice=None,
) -> dict[str, Any]:
    """Standalone PWE solve for an explicitly supplied reciprocal basis.

    Legume's built-in basis is rectangular in reciprocal indices and is not
    C3 closed.  This small adapter reconstructs the Fourier epsilon matrix
    for an arbitrary finite basis, without modifying Legume itself. The
    Hermitian diagonalization uses ``operator + EIGENSOLVER_SHIFT * I`` and
    subtracts that same shift from the squared frequencies; this is the
    documented numerical convention shared by the native and custom paths.
    """

    import legume

    _, layer = build_layer(spec, lattice=lattice)
    gvec = np.asarray(gvec, dtype=float)
    differences = gvec[:, :, None] - gvec[:, None, :]
    eps_matrix = layer.compute_ft(differences.reshape(2, -1)).reshape(gvec.shape[1], gvec.shape[1])
    eps_inv_mat = np.linalg.inv(eps_matrix)
    kpoints = q_to_legume_k(qpoints).T
    frequencies = []
    eigenvectors = []
    for k in kpoints.T:
        kplusg = k[:, None] + gvec
        if pol.lower() == "te":
            matrix = (kplusg.T @ kplusg) * eps_inv_mat
        elif pol.lower() == "tm":
            magnitudes = np.linalg.norm(kplusg, axis=0)
            matrix = np.outer(magnitudes, magnitudes) * eps_inv_mat
        else:
            raise ValueError("polarization must be TE or TM")
        freq2, evecs = np.linalg.eigh(matrix + EIGENSOLVER_SHIFT * np.eye(matrix.shape[0]))
        freq = np.sqrt(np.abs(freq2 - EIGENSOLVER_SHIFT)) / (2.0 * np.pi)
        order = np.argsort(freq)[:numeig]
        frequencies.append(freq[order])
        eigenvectors.append(evecs[:, order])
    return {
        "frequencies": np.asarray(frequencies),
        "eigenvectors": np.asarray(eigenvectors),
        "gvec": gvec,
        "eps_matrix": eps_matrix,
        "eps_inv_mat": eps_inv_mat,
        "kpoints_cartesian": kpoints.T,
        "polarization": pol.lower(),
        "eigensolver_shift": EIGENSOLVER_SHIFT,
        "legume_version": getattr(legume, "__version__", "unknown"),
    }


def solve_pwe_closed_basis(
    spec: GeometrySpec,
    qpoints: np.ndarray,
    *,
    seed_gmax: float,
    closure_qpoints: np.ndarray,
    numeig: int = 4,
    pol: str = "te",
    lattice=None,
) -> dict[str, Any]:
    """Solve with the validated affine-C3 closure of one seed Legume basis."""

    return solve_pwe_point_group_closed_basis(
        spec, qpoints, seed_gmax=seed_gmax, closure_qpoints=closure_qpoints,
        linear_operations=point_group_operations("C3"), numeig=numeig, pol=pol, lattice=lattice,
    )


def solve_pwe_point_group_closed_basis(
    spec: GeometrySpec,
    qpoints: np.ndarray,
    *,
    seed_gmax: float,
    closure_qpoints: np.ndarray,
    linear_operations: tuple[np.ndarray, ...] | list[np.ndarray],
    numeig: int = 4,
    pol: str = "te",
    lattice=None,
) -> dict[str, Any]:
    """Solve with an explicitly verified finite affine point-group closure."""

    from .diagnostics import point_group_orbit_closure
    seed = solve_pwe(spec, closure_qpoints, gmax=seed_gmax, numeig=numeig, pol=pol, lattice=lattice)
    direct_basis = spec.direct_basis if lattice is None else lattice.direct_basis
    closed_gvec = point_group_orbit_closure(
        seed["gvec"], closure_qpoints, list(linear_operations), direct_basis=direct_basis,
    )
    result = solve_pwe_custom_basis(spec, qpoints, closed_gvec, numeig=numeig, pol=pol, lattice=lattice)
    result["seed_gvec_count"] = int(seed["gvec"].shape[1])
    result["closed_gvec_count"] = int(closed_gvec.shape[1])
    result["point_group_operations"] = len(linear_operations)
    return result


def solve_bands(model, qpoints: np.ndarray, *, gmax: float, numeig: int = 4, pol: str = "te") -> dict[str, Any]:
    """Stable Model2D band API; basis policy selects one solver path."""

    if model.basis_policy in {"native", "circular"} or model.point_group is None:
        return solve_pwe(model.geometry, qpoints, gmax=gmax, numeig=numeig, pol=pol, lattice=model.lattice)
    if model.point_group in {"C3", "C4"}:
        operations = point_group_operations(model.point_group)
        if model.closure_qpoints is not None:
            closure_qpoints = model.closure_qpoints
        else:
            seed = np.asarray(qpoints, dtype=float)[0]
            closure_qpoints = np.asarray([operation @ seed for operation in operations])
        return solve_pwe_point_group_closed_basis(
            model.geometry, qpoints, seed_gmax=gmax, closure_qpoints=closure_qpoints,
            linear_operations=operations, numeig=numeig, pol=pol, lattice=model.lattice,
        )
    return solve_pwe(model.geometry, qpoints, gmax=gmax, numeig=numeig, pol=pol, lattice=model.lattice)


def frequency_at_k(model, kpoint: np.ndarray, *, gmax: float, band: int = 0, pol: str = "te") -> float:
    """Return one zero-based band frequency at a single Cartesian q point."""

    result = solve_bands(model, np.asarray(kpoint, dtype=float).reshape(1, 2), gmax=gmax, numeig=max(4, band + 1), pol=pol)
    return float(result["frequencies"][0, band])


def solve_homogeneous_pwe(qpoints: np.ndarray, epsilon: float, *, gmax: float, numeig: int = 4, pol: str = "te") -> dict[str, Any]:
    import legume

    a1, a2 = DIRECT_BASIS.T
    lattice = legume.Lattice(a1, a2)
    layer = legume.ShapesLayer(lattice, eps_b=float(epsilon))
    pwe = legume.PlaneWaveExp(layer, gmax=gmax)
    kpoints = q_to_legume_k(qpoints).T
    pwe.run(kpoints=kpoints, pol=pol.lower(), numeig=numeig)
    return {
        "frequencies": np.asarray(pwe.freqs),
        "gvec": np.asarray(pwe.gvec),
        "kpoints_cartesian": kpoints.T,
        "polarization": pol.lower(),
        "legume_version": getattr(legume, "__version__", "unknown"),
    }


def homogeneous_shell_frequencies(qpoint: np.ndarray, epsilon: float, shell: int = 4) -> np.ndarray:
    reciprocal_no_2pi = np.linalg.inv(DIRECT_BASIS).T
    values = []
    for n1 in range(-shell, shell + 1):
        for n2 in range(-shell, shell + 1):
            cartesian = np.asarray(qpoint, dtype=float) + reciprocal_no_2pi @ np.array([n1, n2], dtype=float)
            values.append(float(np.linalg.norm(cartesian) / math.sqrt(epsilon)))
    return np.sort(np.asarray(values))
