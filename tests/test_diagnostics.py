import numpy as np

from legumephc.diagnostics import projector_distance, rank1_wilson, rankn_wilson


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

