import math
import numpy as np

from legumephc.config import load_benchmark
from legumephc.geometry import area_matched_radius, c3_geometry_residual, geometry_spec, m7_orbit, polygon_vertices, strict_c3_by_construction


def test_frozen_geometry_and_area_matching():
    config = load_benchmark()
    g16, g15, circle = (geometry_spec(config, name) for name in ("G16", "G15", "Circle"))
    assert g16.radii == (0.2, 0.1875)
    assert np.isclose(g15.radii[0], area_matched_radius(0.2, 16, 15))
    assert not strict_c3_by_construction(g16)
    assert strict_c3_by_construction(g15)
    assert strict_c3_by_construction(circle)
    assert g15.epsilon_background == 2.7**2
    assert c3_geometry_residual(g15) < 1e-12
    assert c3_geometry_residual(circle) < 1e-12
    assert c3_geometry_residual(g16) > 1e-3


def test_m7_is_an_independent_c3_orbit():
    orbit = m7_orbit(load_benchmark())
    center = np.array([2 / 3, 0.0])
    radii = np.linalg.norm(orbit - center, axis=1)
    assert orbit.shape == (3, 2)
    assert len({tuple(np.round(point, 14)) for point in orbit}) == 3
    assert np.allclose(radii, 7 / 36)


def test_legume_polygon_edges_are_counter_clockwise():
    vertices = polygon_vertices(0.2, 15, 0.0, np.zeros(2))
    signed_area = 0.5 * np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )
    assert signed_area > 0
