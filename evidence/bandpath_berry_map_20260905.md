# Band path and independent Berry map — 2026-09-05

This report uses only LegumePhC's own public reproducible calculations. The
closed-basis solver is one reusable API (`solve_pwe_closed_basis`): it builds a
single affine-C3-closed reciprocal basis and returns normalized eigenvectors
from the same PWE operator. No MPB, GME, UI, sector copying, averaging,
smoothing, or spike deletion was used.

## Γ–K–M–Γ path

The G15 path uses five independent samples per segment and the first four
bands. Default circular and closed-basis solves were compared at each cutoff.

| gmax | max default-vs-closed path frequency difference | default M7 C3 residual | closed M7 C3 residual | closed basis count |
|---:|---:|---:|---:|---:|
| 4 | 7.097e-4 | 5.429e-4 | 4.656e-10 | 75 (seed 49) |
| 5 | 1.011e-3 | 9.774e-4 | 4.093e-10 | 123 (seed 81) |
| 6 | 6.928e-4 | 4.387e-4 | 4.072e-10 | 183 (seed 121) |

The default path changes by `1.213e-3` then `9.417e-4` between successive
cutoffs; the closed path changes by `1.543e-3` then `7.944e-4`. The closed
M7 pilot gate remains below `1e-6` at all tested cutoffs, so the independent
Berry-map stage was allowed to proceed.

## Independent G15 Berry map

The minimum map is a 5×5 grid spanning ±0.1 in Cartesian reciprocal
coordinates around K=(2/3,0). All 25 grid points and their 120°/240° images
were independently solved, giving 73 unique solved points. The 16 local cells
report the raw rank-2 Wilson phase and phase density, signed area, and a
quality/uncertainty mask. The C3 residual map independently compares each
grid point with both rotated images using the reciprocal-basis projector.

| basis | solved points | raw rank-2 phase-density range | max C3 projector residual | minimum C3 map matched fraction | rank-2 qualified cells | rank-1 status |
|---|---:|---:|---:|---:|---:|---|
| default circular | 73 | 0.892–283.559 | 2.345e-2 | 74.1% | 16/16 | RANK1_WITHHELD |
| closed C3 | 73 | 0.921–279.498 | 4.982e-8 | 100% | 16/16 | RANK1_WITHHELD |

The raw rank-2 map is not replaced by a symmetry average. Its C3 residual map
shows the default-basis asymmetry directly, while the closed-basis residual
falls to numerical precision. Rank-1 is not presented as a band-2 map and is
withheld by the M7 band-gap qualification rule.

The complete immutable map record is
`results/20260905T130906.480809Z-pwe_g15_independent_berry_map-3b02a63a/`.
The band-path record is
`results/20260905T130942.851716Z-pwe_g15_bandpath_default_vs_closed-068d07f3/`.

Plots: [g15_berry_map_default_vs_closed.png](g15_berry_map_default_vs_closed.png)
and [g15_bandpath_default_vs_closed.png](g15_bandpath_default_vs_closed.png).
