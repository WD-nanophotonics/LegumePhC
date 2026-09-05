# Corrected local-plaquette C3 pilot — 2026-09-05

Reproduction:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe run_c3_crosscheck.py --convergence
.venv\Scripts\python.exe run_c3_crosscheck.py --extension
```

The earlier `pwe_c3_convergence` record used a triangle joining the three M7
orbit members. That global path is retained as immutable diagnostic history,
but it is not a local Berry plaquette and its Wilson phase is not used for C3
qualification. The corrected run uses three separate C3-covariant local
plaquettes per geometry/cutoff/step: four corners per member, 12 corners in a
single Legume batch, with the two side vectors rotated by 120 degrees between
members. Every plaquette has signed area `step^2`.

Numerical Wilson qualification is separate from C3 covariance qualification.
The numerical gates are gap > `1e-3`, link magnitude or rank-2 minimum singular
value > `0.1`, and branch margin > `0.1` rad. C3 projector tolerance is data
driven: it is the observed absolute change in the maximum projector residual
between `gmax=4` and `gmax=5`, with no fixed C3 tolerance.

| geometry | cutoff | max frequency residual | max rank-2 projector residual | C3 covariance |
|---|---:|---:|---:|---|
| G15 | 4 | 5.429e-4 | 2.744e-2 | provisional |
| G15 | 5 | 9.774e-4 | 2.114e-2 | C3_NOT_QUALIFIED |
| G15 | 6 | 4.387e-4 | 1.488e-2 | C3_NOT_QUALIFIED |
| Circle | 4 | 5.435e-4 | 2.863e-2 | provisional |
| Circle | 5 | 9.840e-4 | 2.153e-2 | C3_NOT_QUALIFIED |
| Circle | 6 | 4.199e-4 | 1.475e-2 | C3_NOT_QUALIFIED |

At `gmax=6`, the adjacent frequency changes are `5.941e-4` (G15) and
`6.126e-4` (Circle), while the residuals continue to decrease. This makes the
previous `gmax=5` convergence claim provisional rather than final, and the
strict-control extension is the requested discriminating test.

For the finest local step (`0.001`) at `gmax=6`, rank-2 Wilson phase densities
were approximately `(0.04471, 0.04405, 0.04405)` for G15 and
`(0.04741, 0.04675, 0.04675)` for Circle; pairwise C3 residuals were `1.49e-2`
and `1.39e-2`. All three local rank-2 plaquettes were numerically qualified;
all three local rank-1 plaquettes were `RANK1_WITHHELD`. Rank-1 link and branch
values remain recorded but are never promoted when the gap gate fails.

The composite `[2,3]` scalar density is the sum of the two orthonormal-mode
densities. Its rotated residual at `gmax=6`, finest cutoff, was `8.48e-3`
(G15) and `8.33e-3` (Circle) at both grid sizes 32 and 64, separating grid
mapping error from band-subspace mixing. The same composite-density and
rank-2 Wilson invariance under U(2) mixing is covered by unit tests.

Complete corrected records are in the immutable local results directories
`results/20260905T123412.173780Z-pwe_c3_convergence-2ab0c4ad/` and
`results/20260905T123518.992414Z-pwe_c3_extension-0b35d60c/`. Diagnostic plot:
[g15_gmax6_frequencies.png](g15_gmax6_frequencies.png).
