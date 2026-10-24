# Valley Chern number over the K half-BZ (MePhC E9F)

## Definition (legacy contract)

C_v^K is the integral of rank-1 Berry curvature over the half Brillouin zone
around K, on a 1/36 reciprocal grid with declared excluded cells, using
local-plaquette Wilson phases (no sector copying, no symmetry expansion).
Qualification uses an absolute gap threshold of **0.02**. E9F classified this
threshold as `HISTORICAL_ENGINEERING_POLICY`, not
`MATHEMATICALLY_REQUIRED_EXACT_0P02` and not
`EMPIRICALLY_VALIDATED_SAFETY_MARGIN`.

## Result (frozen endpoint, MPB, work order `MEPHC-E9F-C2-QP-A-20260826-276`)

| Band (0-based) | C_v^K | Coverage | Status |
|---|---:|---:|---|
| 0 | **−0.09405797** | 551/551 | REPORTED |
| 1 | **+0.50869157** | 551/551 | REPORTED |
| 2 | — | 534/551 (17 failed) | `INCOMPLETE_NOT_REPORTED`, no numeric Chern |

- A composite band2+band3 value was `NOT_AUTHORIZED`.
- The FR04 rank-2/rank-3 composite method was classified `NOT_ESTABLISHED`
  (`d10_fr04_composite_method_validation_result.json`).
- E9F's stated next step was a threshold-validation contract, not more solves.

## Stencil findings (E9F)

- Finite-stencil and boundary-stencil artefacts were supported as real
  contributors near gap-closing regions.
- **Monotone stencil contraction is not a valid general convergence
  requirement.** Shrinking the plaquette does not have to change the result
  monotonically.

## For LegumePhC

Ported on 2026-10-24 as `legumephc.solve_valley_chern` (Studio study type
"Valley Chern (half BZ)"). It reproduces this domain and both estimators
exactly, keeps the gap/link/branch gates with 0.02 as a reported parameter,
and reports coverage next to every value. FR0 results and the comparison with
the values above are in `evidence/valley_chern_dai_fr0_20261024.md`.

## Provenance

- `MPW:audit/e9f/a_source_valley_chern_contract.json`
- `MPW:audit/e9f/qp_a_science_decision.json`
- `MPW:audit/e9f/rp3_a_r128_science_seal.json`
- `MPW:audit/e9f/d8_fr04_rank2_pair12_qualification_berry.json`
- `MPW:audit/e9f/d10_fr04_composite_method_validation_result.json`
- Locked dataset: `WSL:.local/share/mephc-runtime/science/MEPHC/E9F_QP_B_C2_C3_R8_LOCKED_SET` (kept)
