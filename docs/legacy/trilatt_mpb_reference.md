# TriLatt MPB band reference (MePhC-TriLatt)

These are MPB TE band frequencies (ωa/2πc) for the two-hole triangular
lattice used across MePhC.

Each case has:
- lattice constant a = 400 nm
- background n_eff = 2.7, with air holes
- two regular-polygon holes per cell: radius r1 at orientation 0° and r2 at
  60°, with n sides each

Height 100 nm was recorded but does not enter the 2D solve. Values are read
from the stored path records at the Γ, K and M ticks.

| Geometry ID | Polygon | MPB res | Γ (bands 1–3) | K | M |
|---|---|---:|---|---|---|
| `a400_r75-75_n16-16_t0-60_neff2p7_h100` | 16-gon, r=75/75 | 32 | 0, 0.44314, 0.46558 | 0.28840, 0.28847, 0.31438 | 0.24191, 0.27678, 0.40080 |
| `a400_r80-75_n16-16_t0-60_neff2p7_h100` | 16-gon, r=80/75 | 32 | 0, 0.44531, 0.47109 | 0.28988, 0.29444, 0.31918 | 0.24450, 0.28162, 0.40489 |
| `a400_r115-115_n3-3_t0-60_neff2p7_h100` | triangle, r=115/115 | 64 | 0, 0.46482, 0.47007 | 0.28490, 0.28536, 0.33901 | 0.23932, 0.28669, 0.41273 |
| `a400_r120-110_n3-3_t0-60_neff2p7_h100` | triangle, r=120/110 | 64 | 0, 0.46138, 0.47142 | 0.28269, 0.28641, 0.33723 | 0.23868, 0.28533, 0.41190 |

## Notes

- Equal radii give a near-degenerate Dirac point at K: 0.28840 vs. 0.28847,
  and 0.28490 vs. 0.28536. Unequal radii open a valley gap: Δ ≈ 0.0046 for
  r80/75 and 0.0037 for r120/110. This is the gap the valley-Chern work
  relies on.
- The r80/75 16-gon case is the geometry behind the C3 investigation
  ([mpb_c3_defect.md](mpb_c3_defect.md)).
- The MPB resolutions here are low (32/64), and M45 showed that C3-level
  agreement needs more than R256. Treat these values as a ~1e-3 check, not a
  1e-5 check.

## Provenance

`WSL:TriLatt/data/<geometry id>/band_nb3_gkm.pkl`; Berry and EFS pickles sit
in the same folders. The TriLatt workspace is a git repo
(`WD-nanophotonics/MePhC-TriLatt`) with uncommitted work, and it is not part
of the cleanup.
