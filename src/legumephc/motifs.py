"""Compact triangular / honeycomb parameters shared by scripts and Studio."""
import numpy as np
from .geometry import Lattice2D, SUBLATTICE_CENTERS


def triangular_motifs(radius, angle=None, *, kind="circle", sides=16, epsilon=1.0, scale=1.0):
    radii = np.atleast_1d(radius).astype(float)
    if radii.ndim != 1 or len(radii) not in (1, 2) or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Radius requires one or two positive numbers")
    angles = np.zeros(len(radii)) if angle is None else np.atleast_1d(angle).astype(float)
    if angles.shape != radii.shape or not np.isfinite(angles).all():
        raise ValueError("Rotation requires one value per radius")
    if kind not in {"circle", "polygon"} or (kind == "polygon" and (int(sides) != sides or sides < 3)):
        raise ValueError("Invalid motif shape or polygon sides")
    centre = Lattice2D.triangular(scale).direct_basis.sum(axis=1) / 2
    centers = centre[None, :] if len(radii) == 1 else (SUBLATTICE_CENTERS - SUBLATTICE_CENTERS.mean(axis=0)) * scale + centre
    return [{"name": f"{'Honeycomb' if len(radii) == 2 else 'Triangular'} {kind} {i + 1}", "kind": kind, "radius": float(r), "angle_degrees": float(angles[i]) if kind == "polygon" else 0.0, "sides": int(sides) if kind == "polygon" else 0, "epsilon": float(epsilon), "center": centers[i].tolist()} for i, r in enumerate(radii)]
