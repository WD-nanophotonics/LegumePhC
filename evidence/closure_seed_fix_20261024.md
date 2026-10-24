# Default closed-basis seed fix — 2026-10-24

## Problem

For models with a verified C3/C4 point group and no explicit
`Model2D.closure_qpoints`, `solve_bands` closed the plane-wave basis on the
rotation orbit of **the first requested q point**.
`point_group_orbit_closure` rounds q_{i+1} − R·q_i to the nearest
reciprocal-lattice vector, which is exact only when the seed is a point-group
fixed point (Γ, K, K′, M). For a generic first point the rounded shifts are
arbitrary, so the basis — and every result computed with it — depended on
which q came first. This affected grid-sampled operations on symmetric
models: first-BZ Berry maps, EFS grids, band paths only by accident of
ordering, and `frequency_at_k` at generic points.

Demonstrated in `tests/test_closure_seed.py::test_old_default_depended_on_q_order_new_one_does_not`:
the pre-fix rule gives different bases for a random permutation of the same
first-BZ plaquette corners.

## Fix

`geometry.closure_center` picks an exact fixed point (zero rounding):
the fixed point nearest the requested q set when every point lies within half
the fixed-point spacing of it (orbits or plaquettes around K, points near M,
…), otherwise Γ (paths, grids, half-BZ domains), giving a pure rotation
closure. `solve_bands` closes on that point's orbit and returns the
`closure_qpoints` it used. Explicit `closure_qpoints` (benchmark models,
`run_c3_crosscheck.py`) are untouched.

## Verified

* Basis identical (as integer reciprocal indices) under any q permutation;
  first-BZ Berry curvature permutes exactly with its plaquettes; EFS-grid
  basis is closed under the 120° rotation.
* Orbits around K (the M7 use case) get **exactly** the historical basis, so
  committed C3 evidence and benchmark studies are unchanged.
* C4: local requests near M close on M; paths close on Γ.
* Valley Chern (`evidence/valley_chern_dai_fr0_20261024.md`) unchanged
  bit-for-bit (it already closed on Γ; its special case is now removed).
* Full suite 115 passed; acceptance matrix 18/18.

## Impact on earlier results

* The user's existing Studio project (`projects/Untitled…`) has no verified
  point group (its honeycomb sites are not C3 about the origin), so it always
  used the native basis and is **not affected**.
* For a C3 model's first-BZ Berry map (Dai FR0, 4×4 grid, band 1) the old
  arbitrary shifts produced a *larger* basis at the same gmax (141 vs. 73 at
  gmax 4; 287 vs. 181 at gmax 6). Against a gmax 12 reference, accuracy
  tracks basis size, not the seed: old N_G 287 → 6.2 % rms error, new
  N_G 337 → 5.4 %. Old maps were order-dependent, not systematically wrong.
* Consequence: at a given gmax the default basis is now smaller. Judge
  convergence by the plane-wave count (`closed_gvec_count`, stored in every
  record), not by gmax.
