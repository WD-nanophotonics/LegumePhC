# Dai FR0 valley Chern numbers over the truncated half BZ — 2026-10-24

First LegumePhC computation of the valley Chern number C_v, carried over
from MePhC E9F (`docs/legacy/valley_chern_e9f.md`). Operation:
`legumephc.solve_valley_chern`; study: `studies/valley_dai/study.py`
(immutable records under `studies/valley_dai/results/`, not committed).

## Model

| | |
|---|---|
| Lattice | triangular, LegumePhC basis a₁=(½, √3/2), a₂=(½, −√3/2) |
| Hole | air equilateral triangle, circumradius 0.4a, one vertex on +y (LegumePhC `angle_degrees=0`; identical to MePhC E9 `physical_vertices`) |
| Background | ε = 7.0225 (n = 2.65), TE |
| Symmetry | point group **C3 verified**; `basis_policy=auto` → C3-closed plane-wave basis |
| Basis seed | C3 orbit of Γ (pure rotation closure; identical basis for K and K′ and every q) |

K-point TE frequencies converge from below toward the legacy MPB R96 values
(gmax 14, 937 plane waves: 0.26749 / 0.31231 / 0.35263 / 0.56254 vs. MPB R96
0.26783 / 0.31245 / 0.35446 / 0.56293), confirming geometry, orientation and
polarization.

## Domain and quadrature (identical to legacy E9F)

* K′ half BZ = triangle (Γ, −b₁, −b₂) centred on K′=(−2/3, 0), shrunk by
  δK = 0.10 (circumradius decrement); three Γ-centred regular hexagons of
  circumradius δΓ = 0.10 (one vertex on +qₓ) excluded.
* Grid q = (i, j)/36. **Source grid** (primary, paper/legacy rule): 551 nodes
  inside the truncated domain, weight 1/36² each (Σw = 0.425154).
  **Clipped cell** (cross-check): 601 cells, exact clipped areas
  (Σw = 0.417136 = continuous area). Both counts and weights equal legacy
  `c1_validation_result.json` to 1e-12 (unit test).
* Ω on a counter-clockwise square plaquette of side 1/36 centred on each node;
  C_v = Σ Ω_q w_q / 2π (q reduced, no extra Jacobian).
* A band is reported only if every sample passes gap ≥ 0.02 (legacy policy),
  link > 0.1 and branch margin > 0.1; otherwise `INCOMPLETE_NOT_REPORTED`
  with no number.

## Sign convention (important)

LegumePhC's Wilson curvature (`rank1_wilson`: +arg Π⟨uᵢ|uᵢ₊₁⟩ / area on a
CCW loop) equals **−Ω of Xiao–Chang–Niu** (A = i⟨u|∇u⟩); verified on a
massive Dirac model to 1e-6 (`tests/test_valley.py`). MePhC's Berry code uses
the same formula. Values below are in the LegumePhC convention; the
textbook-sign value is the negative.

## Results — K′ half BZ, LegumePhC sign, source-grid estimator

| gmax | plane waves | band 0 | band 1 | band 2 |
|---:|---:|---:|---:|---|
| 4 | 73 | +0.0629 | −0.5941 | not reported (537/551; gap) |
| 5 | 121 | +0.0695 | −0.5762 | not reported (537/551) |
| 6 | 181 | +0.0736 | −0.5658 | not reported (535/551) |
| 7 | 253 | +0.0763 | −0.5587 | not reported (535/551) |
| 8.5 | 337 | +0.0783 | −0.5539 | not reported (535/551) |
| 9.5 | 433 | +0.0798 | −0.5503 | not reported (535/551) |
| 10.5 | 541 | +0.0811 | −0.5474 | not reported (535/551) |
| 12 | 661 | +0.0820 | −0.5452 | not reported (535/551) |

* **All 551 samples qualify for bands 0 and 1 at every cutoff** (min gaps
  0.046 and 0.028–0.037 ≥ 0.02). Band 2 never qualifies: its minimum gap is
  0.0056 (near M), so 16 samples fail the 0.02 gate and no number is
  reported — the same failure as legacy E9F (534/551, 17 failed).
* **Clipped-cell estimator** agrees within 0.0001 (band 0) and 0.0026 (band 1).
* **Wilson stencil** (gmax 6): 1/36 → 1/72 → 1/144 changes band 0 by <1e-5
  and band 1 by 7e-4 — the stencil is converged; the cutoff is not.
* **Time reversal** (gmax 6): K gives −0.0736 / +0.5658, exactly −C(K′).

### Cutoff extrapolation (indicative only)

C(N_G) fits C∞ + A/√N_G over the last 4–5 cutoffs with residuals ≤ 6e-6:
**C∞ ≈ +0.091 (band 0), −0.523 (band 1)**. A 1/gmax fit gives +0.090 /
−0.525 and a 1/N^0.75 fit +0.087 / −0.532, so the model uncertainty is about
±0.004 / ±0.009. This is an extrapolation, not a converged value:
`convergence_status = NOT_ASSESSED` in every record.

## Comparison (textbook sign = −LegumePhC sign)

| band (0-based) | LegumePhC gmax 12 | LegumePhC extrapolated | legacy MePhC E9F (MPB R64) | Dai paper (paper K) |
|---|---:|---:|---:|---:|
| 0 | −0.082 | −0.091 ± 0.004 | −0.094 | −0.10 |
| 1 | +0.545 | +0.523 ± 0.009 | +0.509 | +0.54 |
| 2 | not reported | — | not reported (534/551) | −0.43 |

In the textbook convention LegumePhC's K′ values have the same signs as legacy
MePhC (whose Berry code uses the *same* formula as LegumePhC) and as the
paper's K anchors. So the MPB pipeline carries one extra sign relative to
LegumePhC at the identical q domain. Its origin is **not localized**: LegumePhC
builds the operator directly in Cartesian (k+G) with a standard e^{−iG·r}
Fourier transform (checked) and Bloch factor e^{+ik·r}; MPB was given the same
left-handed direct basis (det = −√3/2), whose handling inside MPB cannot be
checked without running MPB. Until resolved, compare magnitudes across
pipelines, not signs.

Magnitudes agree with legacy within 3% (band 0) and 3% (band 1); legacy was
itself an unconverged MPB R64 result. Both bands sum to ≈ +0.43 (textbook),
consistent with the paper's statement that the first three bands sum to ≈ 0
only once band 2 (−0.43 in the paper) is included — which neither LegumePhC
nor MePhC can qualify under the 0.02 gap gate.

## Uncertainty and open items

* Cutoff convergence: monotone but slow (≈ 1/√N_G); a qualified value needs
  either larger bases or a pre-registered extrapolation rule.
* Inner product: LegumePhC uses the TE plane-wave H coefficients (exact
  Hermitian form for μ = 1); legacy E9 used MPB's energy-weighted E+H overlap.
* Sign: one unexplained sign between the MPB and LegumePhC pipelines (above).
* `solve_bands`' default closure seed (orbit of the first q point) is only
  correct for high-symmetry orbits; `solve_valley_chern` seeds at Γ. Other
  grid-sampled operations (e.g. Studio first-BZ Berry maps) should be
  reviewed for the same issue.
* FR0.4 (rounded triangle) needs an arbitrary-polygon motif, not yet in
  `GeometrySpec`.

Summary data: `evidence/valley_chern_dai_fr0_20261024_summary.json`.
