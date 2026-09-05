import numpy as np

from legumephc.diagnostics import (
    c3_closed_reciprocal_basis,
    projector_distance,
    rank1_wilson,
    rankn_wilson,
    reciprocal_c3_map,
    rotated_density_residual,
    scalar_field_density,
)
from legumephc.geometry import DIRECT_BASIS, rotation


def _random_subspaces(seed=9):
    rng = np.random.default_rng(seed)
    return [np.linalg.qr(rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2)))[0] for _ in range(4)]


def test_rank1_wilson_is_u1_invariant():
    rng = np.random.default_rng(4)
    vectors = [rng.normal(size=7) + 1j * rng.normal(size=7) for _ in range(4)]
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=4))
    assert np.isclose(rank1_wilson(vectors)["phase"], rank1_wilson([p * v for p, v in zip(phases, vectors)])["phase"])


def test_rankn_wilson_and_projector_are_u2_invariant():
    rng = np.random.default_rng(3)
    spaces = _random_subspaces()
    rotations = [np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0] for _ in spaces]
    transformed = [space @ unitary for space, unitary in zip(spaces, rotations)]
    assert np.isclose(rankn_wilson(spaces)["phase"], rankn_wilson(transformed)["phase"])
    assert np.isclose(projector_distance(spaces[0], spaces[1]), projector_distance(transformed[0], transformed[1]))


def test_composite_density_is_invariant_under_u2_band_mixing():
    rng = np.random.default_rng(12)
    vectors = np.stack([
        np.linalg.qr(rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4)))[0]
        for _ in range(3)
    ])
    unitary = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    mixed = vectors.copy()
    mixed[:, :, 1:3] = np.einsum("kgb,bc->kgc", vectors[:, :, 1:3], unitary)
    reciprocal = np.linalg.inv(DIRECT_BASIS).T
    gvec = 2 * np.pi * (reciprocal @ np.array([[0, 1, -1, 0, 1], [0, 0, 0, 1, -1]], dtype=float))
    qpoints = np.array([[0.2, 0.1], [0.3, -0.1], [-0.1, 0.2]])
    original = scalar_field_density(vectors, gvec, qpoints, basis=DIRECT_BASIS, grid_size=16)
    remixed = scalar_field_density(mixed, gvec, qpoints, basis=DIRECT_BASIS, grid_size=16)
    assert np.allclose(original[..., 1:3].sum(axis=-1), remixed[..., 1:3].sum(axis=-1))


def test_rotated_density_residual_uses_periodic_cell_coordinates():
    size = 24
    fractional = np.stack(np.meshgrid(
        np.arange(size) / size,
        np.arange(size) / size,
        indexing="xy",
    ), axis=-1)
    base = 1.0 + 0.2 * np.cos(2 * np.pi * fractional[..., 0]) + 0.1 * np.sin(4 * np.pi * fractional[..., 1])
    transform = rotation(120.0)
    cart = fractional.reshape(-1, 2) @ DIRECT_BASIS.T
    inverse_fractional = (cart @ transform) @ np.linalg.inv(DIRECT_BASIS).T
    indices = np.rint((inverse_fractional % 1.0) * size).astype(int) % size
    densities = [base]
    for _ in range(2):
        densities.append(densities[-1][indices[:, 1].reshape(size, size), indices[:, 0].reshape(size, size)])
    values = rotated_density_residual(
        np.asarray(densities)[..., None], np.zeros((3, 2)),
        basis=DIRECT_BASIS, rotation_matrix=transform,
    )
    assert values.shape == (3, 1)
    assert np.max(values) < 1e-12


def test_reciprocal_c3_map_includes_m7_reciprocal_shift():
    q0 = np.array([0.4722222222222222, 0.0])
    q1 = np.array([0.7638888888888888, -0.16839383310241256])
    source = 2 * np.pi * np.array([[0.0, 1.0], [0.0, 0.0]])
    transform = rotation(120.0)
    shift = q1 - transform @ q0
    target = 2 * np.pi * (transform @ (source / (2 * np.pi)) - shift[:, None])
    result = reciprocal_c3_map(
        source, target, q0, q1, rotation_matrix=transform,
    )
    assert result["matched_count"] == 2
    assert result["matched_fraction"] == 1.0
    assert np.allclose(result["shift_reciprocal"], [1.0, -1 / np.sqrt(3)])
    assert result["maximum_matching_residual"] < 1e-12


def test_closed_reciprocal_basis_is_permuted_by_each_c3_step():
    q0 = np.array([0.4722222222222222, 0.0])
    q1 = np.array([0.7638888888888888, -0.16839383310241256])
    q2 = np.array([0.7638888888888888, 0.16839383310241256])
    qpoints = np.asarray([q0, q1, q2])
    reciprocal = np.linalg.inv(DIRECT_BASIS).T
    seed = 2 * np.pi * reciprocal @ np.array([
        [-1, 0, 1, -1, 0, 1],
        [-1, -1, -1, 0, 0, 0],
    ], dtype=float)
    closed = c3_closed_reciprocal_basis(seed, qpoints, rotation_matrix=rotation(120.0))
    for index in range(3):
        mapping = reciprocal_c3_map(
            closed, closed, qpoints[index], qpoints[(index + 1) % 3],
            rotation_matrix=rotation(120.0),
        )
        assert mapping["matched_count"] == closed.shape[1]
        assert len(set(mapping["indices"])) == closed.shape[1]
