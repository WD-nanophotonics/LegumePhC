# Closed-basis Berry and scalar validation — 2026-09-05

This experiment uses the same standalone affine-C3-closed reciprocal basis
adapter as the causal diagnosis. The Legume site-packages are unchanged. The
gmax=5 default seed has 81 vectors; its exact integer-coordinate C3 closure has
123 vectors. Each of the three M7 members uses a separate four-corner local
plaquette, with 12 corners per step solved in one batch. No sector averaging,
copying, smoothing, or raw complex-field comparison is used.

Numerical qualification remains separate from C3 covariance. Every local
rank-2 plaquette passes the gap/link/branch gates. Every local rank-1 result is
`RANK1_WITHHELD`: the band-2 gap is about `6.55e-4`, below the `1e-3` gate,
despite acceptable links and branch margins. Association is explicitly checked
through the adjacent gauge-invariant link magnitude (rank 1) or minimum
singular value (rank 2).

At the finest step `0.001`, the default gmax=7 pairwise rank-2 phase-density
residuals were `5.884e-3` (G15) and `6.457e-3` (Circle). With the closed basis
at the same local-plaquette calculation, they were `5.475e-8` and
`1.391e-8`; the corresponding rank-2 minimum link singular values were
`0.999999` for all three members. Thus the Berry asymmetry disappears to
numerical precision when the reciprocal basis is closed, while rank-1 remains
withheld rather than being promoted.

| case | H scalar residual, band 2 | E scalar residual, band 2 | energy scalar residual, band 2 | composite [2,3] H residual |
|---|---:|---:|---:|---:|
| G15 closed | 4.491e-8 | 2.587e-8 | 4.281e-8 | 4.577e-9 |
| Circle closed | 2.387e-12 | 1.457e-12 | 2.289e-12 | 3.116e-14 |
| G16 closed | 1.408e-4 | 1.638e-4 | 1.347e-4 | 8.051e-6 |

For closed G15, the finest-step rank-2 phase-density triple is
`(0.044715, 0.044049, 0.044047)` with pairwise residual `5.475e-8`. Circle is
`(0.047407, 0.046748, 0.046746)` with pairwise residual `1.391e-8`. G16 is
`(0.044020, 0.044019, 0.044019)` with pairwise residual `3.394e-5`; it remains
an approximate-C3 control, not an exact pass.

The first broken layer is therefore causally identified as the default
reciprocal-basis truncation. Frequencies, H/E/energy scalars, composite
density, rank-2 projectors, and rank-2 local Berry covariance all recover when
the finite basis is made affine-C3 closed. This does not establish blanket MPB
equivalence: no public compact MPB record was available for a matched E/H,
frequency, projector, and Berry comparison, so those MPB claims remain
unqualified.

Complete record:
`results/20260905T130153.610802Z-pwe_c3_closed_local_plaquette-5f126ff3/`.
