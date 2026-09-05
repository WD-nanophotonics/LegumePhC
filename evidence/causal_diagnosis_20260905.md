# Reciprocal-cut causal diagnosis — 2026-09-05

The exact C3 plane-wave map used here is
`R(q + G) = q_next + G_next`, with
`G_next = R G - (q_next - R q)`. For the M7 orbit,
`q_next - R q = R*K - K` is a reciprocal-lattice vector. Testing `R G`
without this shift would be the wrong little-group operation.

The actual circular Legume `gvec` set does not close under this map for either
strict control. The maximum matching residual among matched vectors is below
`2.0e-15` in every case; the non-closure is entirely a finite-cutoff boundary
effect, not a numerical matching failure. Because no gmax=2..6 set closes, the
full dielectric-operator covariance residual is not defined on the truncated
matrix and is reported as `null` rather than inferred from an invalid
permutation.

| geometry | gmax | matched fraction | unmatched vectors | max frequency C3 residual | rank-2 projector residual |
|---|---:|---:|---:|---:|---:|
| G15 | 2 | 66.7% | 3 | 1.203e-2 | 1.509e-1 |
| G15 | 3 | 72.0% | 7 | 6.546e-3 | 7.939e-2 |
| G15 | 4 | 73.5% | 13 | 5.429e-4 | 2.744e-2 |
| G15 | 5 | 74.1% | 21 | 9.774e-4 | 2.114e-2 |
| G15 | 6 | 74.4% | 31 | 4.387e-4 | 1.488e-2 |
| Circle | 2 | 66.7% | 3 | 1.237e-2 | 1.547e-1 |
| Circle | 3 | 72.0% | 7 | 6.887e-3 | 8.029e-2 |
| Circle | 4 | 73.5% | 13 | 5.435e-4 | 2.863e-2 |
| Circle | 5 | 74.1% | 21 | 9.840e-4 | 2.153e-2 |
| Circle | 6 | 74.4% | 31 | 4.199e-4 | 1.475e-2 |

The missing-boundary fraction decreases from 33.3% to 25.6% while frequency
and projector residuals co-decrease. This supports cutoff-boundary truncation
as the causal explanation for the residuals. Operator covariance was correctly
not claimed because the needed permutation is incomplete.

The minimum strict-control extension was then run only for G15 and Circle at
`gmax=7`, reusing the corrected 12-corner local plaquette and composite
`[2,3]` density diagnostics:

| geometry | gmax=6→7 max frequency delta | gmax=7 frequency residual | projector residual | composite density residual (32=64) |
|---|---:|---:|---:|---:|
| G15 | 5.421e-4 | 3.865e-4 | 1.201e-2 | 6.709e-3 |
| Circle | 5.846e-4 | 4.252e-4 | 1.232e-2 | 6.938e-3 |

At `gmax=7`, finest-step rank-2 phase-density triples were
`(0.043511, 0.043255, 0.043255)` for G15 and
`(0.046207, 0.045908, 0.045909)` for Circle; pairwise residuals were
`5.884e-3` and `6.457e-3`. All local rank-2 numerical gates passed; rank-1
remained `RANK1_WITHHELD`. C3 covariance remains a separate
`C3_NOT_QUALIFIED` conclusion, not a failure of Legume as a solver.

Complete records:

- `results/20260905T124117.824695Z-pwe_reciprocal_c3_operator_diagnosis-6b1ecdcc/`
- `results/20260905T124226.982994Z-pwe_c3_gmax7-471c19a5/`

Diagnostic plot: [g15_gmax7_frequencies.png](g15_gmax7_frequencies.png).
