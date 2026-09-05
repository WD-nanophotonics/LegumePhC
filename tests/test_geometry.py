import math
import numpy as np

from legumephc.config import load_benchmark
from legumephc.geometry import (
    Affine2D,
    GeometrySpec,
    Lattice2D,
    area_matched_radius,
    c3_geometry_residual,
    first_bz_labels,
    first_bz_vertices,
    geometry_spec,
    identity_path,
    m7_orbit,
    point_group_operations,
    point_group_geometry_residual,
    polygon_vertices,
    square_circle_spec,
    strict_c3_by_construction,
)
from legumephc.model import Model2D


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


def test_lattice_paths_and_generic_affine_labels_are_honest():
    labels, points = identity_path(Lattice2D.triangular(), samples_per_segment=2)
    assert labels == ("Gamma", "K", "M", "Gamma")
    assert points.shape == (7, 2)
    labels, points = identity_path(Lattice2D.square(), samples_per_segment=2)
    assert labels == ("Gamma", "X", "M", "Gamma")
    assert points.shape == (7, 2)
    config = load_benchmark()
    generic = Model2D.from_benchmark(
        config, "G15", lattice=Lattice2D(Lattice2D.triangular().direct_basis, kind="custom"),
        affine=Affine2D(linear=np.array([[1.0, 0.2], [0.0, 1.0]])),
    )
    assert generic.point_group is None
    assert first_bz_labels(generic.lattice) == ("Gamma", "P1", "P2", "Gamma")
    bz = first_bz_vertices(generic.lattice)
    _, generic_path = identity_path(generic.lattice, samples_per_segment=2)
    assert bz.shape[0] >= 4
    assert np.isclose(np.linalg.norm(generic_path[-1]), 0.0)
    assert len(point_group_operations("C3")) == 3
    assert len(point_group_operations("C4")) == 4
    square = Lattice2D.square()
    assert point_group_geometry_residual(square_circle_spec(), square, "C4") < 1e-12
    assert Model2D(square_circle_spec(), square).point_group == "C4"
    assert Model2D(geometry_spec(config, "G15"), square, unverified_point_group="C4").point_group is None
    assert Model2D.from_benchmark(config, "G15", affine=Affine2D(linear=np.array([[1.0, 0.2], [0.0, 1.0]]))).point_group is None
    distorted = Model2D(square_circle_spec(), square, affine=Affine2D(linear=np.array([[1.0, 0.2], [0.0, 1.0]])))
    assert distorted.point_group is None
    assert distorted.effective_lattice.kind == "custom"
    assert np.allclose(distorted.effective_lattice.direct_basis, distorted.affine.linear @ square.direct_basis)
    effective = distorted.effective_geometry
    assert np.allclose(effective.centers[0], distorted.affine.apply(square_circle_spec().centers)[0])
    assert effective.ellipse_parameters[0] is not None
    singular_values = np.linalg.svd(distorted.affine.linear, compute_uv=False)
    assert np.allclose(effective.ellipse_parameters[0][:2], square_circle_spec().radii[0] * singular_values)
    from legumephc.observables import _motif_mask
    transformed_center = effective.centers[0]
    inside = transformed_center + distorted.affine.linear @ np.array([0.9 * square_circle_spec().radii[0], 0.0])
    outside = transformed_center + distorted.affine.linear @ np.array([1.1 * square_circle_spec().radii[0], 0.0])
    assert _motif_mask(np.vstack((inside, outside)), effective, distorted.effective_lattice.direct_basis, 0).tolist() == [True, False]
    polygon = GeometrySpec(
        name="SquarePolygon", kind="polygon", radii=(0.1,), sides=(4,), angles_degrees=(0.0,),
        strict_c3=False, epsilon_background=2.0, epsilon_inclusion=5.0,
        direct_basis=np.eye(2), centers=np.array([[0.3, 0.4]]),
    )
    polygon_model = Model2D(polygon, square, affine=distorted.affine)
    source_vertices = polygon_vertices(0.1, 4, 0.0, polygon.centers[0])
    assert np.allclose(polygon_model.effective_geometry.transformed_vertices[0], polygon_model.affine.apply(source_vertices))
