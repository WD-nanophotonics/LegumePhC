import numpy as np

from legumephc.diagnostics import (
    projector_distance,
    rank1_wilson,
    rankn_wilson,
    rotated_density_residual,
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
