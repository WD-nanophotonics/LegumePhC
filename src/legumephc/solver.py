from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import DIRECT_BASIS, SUBLATTICE_CENTERS, GeometrySpec, polygon_vertices


def build_layer(spec: GeometrySpec):
    import legume

    a1, a2 = DIRECT_BASIS.T
    lattice = legume.Lattice(a1, a2)
    layer = legume.ShapesLayer(lattice, eps_b=spec.epsilon_background)
    for index, (center, radius) in enumerate(zip(SUBLATTICE_CENTERS, spec.radii)):
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


def solve_pwe(spec: GeometrySpec, qpoints: np.ndarray, *, gmax: float, numeig: int = 4, pol: str = "te") -> dict[str, Any]:
    import legume

    _, layer = build_layer(spec)
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
        "legume_version": getattr(legume, "__version__", "unknown"),
    }


def solve_pwe_custom_basis(
    spec: GeometrySpec,
    qpoints: np.ndarray,
    gvec: np.ndarray,
    *,
    numeig: int = 4,
    pol: str = "te",
) -> dict[str, Any]:
    """Standalone PWE solve for an explicitly supplied reciprocal basis.

    Legume's built-in basis is rectangular in reciprocal indices and is not
    C3 closed.  This small adapter reconstructs the Fourier epsilon matrix
    for an arbitrary finite basis, without modifying Legume itself.
    """

    import legume

    _, layer = build_layer(spec)
    gvec = np.asarray(gvec, dtype=float)
    differences = gvec[:, :, None] - gvec[:, None, :]
    eps_matrix = layer.compute_ft(differences.reshape(2, -1)).reshape(gvec.shape[1], gvec.shape[1])
    eps_inv_mat = np.linalg.inv(eps_matrix)
    kpoints = q_to_legume_k(qpoints).T
    frequencies = []
    eigenvectors = []
    for k in kpoints.T:
        if pol.lower() != "te":
            raise ValueError("the custom adapter currently supports TE only")
        kplusg = k[:, None] + gvec
        matrix = (kplusg.T @ kplusg) * eps_inv_mat
        freq2, evecs = np.linalg.eigh(matrix + np.eye(matrix.shape[0]))
        freq = np.sqrt(np.abs(freq2 - 1.0)) / (2.0 * np.pi)
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
) -> dict[str, Any]:
    """Solve with the validated affine-C3 closure of one seed Legume basis."""

    from .diagnostics import c3_closed_reciprocal_basis
    seed = solve_pwe(spec, closure_qpoints, gmax=seed_gmax, numeig=numeig, pol=pol)
    closed_gvec = c3_closed_reciprocal_basis(
        seed["gvec"], closure_qpoints,
        rotation_matrix=np.array([[np.cos(2 * np.pi / 3), -np.sin(2 * np.pi / 3)],
                                  [np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)]]),
    )
    result = solve_pwe_custom_basis(spec, qpoints, closed_gvec, numeig=numeig, pol=pol)
    result["seed_gvec_count"] = int(seed["gvec"].shape[1])
    result["closed_gvec_count"] = int(closed_gvec.shape[1])
    return result


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
