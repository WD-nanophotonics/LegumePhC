# Dai triangular-hole paper reproduction attempt (MePhC E7I5, E9A–E9E)

## Model, human-corrected contract (E9B/E9C)

| Parameter | Value |
|---|---|
| Lattice | triangular, a = 400 nm |
| Hole | equilateral triangle, circumradius 0.4a (160 nm), side 0.6928a, orientation 90° mod 120° |
| Fill fraction | 0.24 (FR0) |
| Background | n = 2.65 (ε = 7.0225); holes are air |
| Polarisation | TE |
| MPB | mesh 3, tolerance 1e-7, R64 with R96 anchor |
| Public K, M | K = (2/3, 0), M = (1/2, 0.288675) in reduced Cartesian coordinates |

## Band frequencies (ωa/2πc), MPB

| Point | R64 | R96 |
|---|---|---|
| Γ | 0, 0.44983, 0.49562, 0.49569 | 0, 0.44928, 0.49419, 0.49419 |
| K | 0.26833, 0.31348, 0.35633, 0.56447 | 0.26783, 0.31245, 0.35446, 0.56293 |
| M | 0.23481, 0.30583, 0.44361, 0.47404 | 0.23414, 0.30405, 0.44242, 0.47289 |

E9A ran a K-only spectrum at R64 with a different geometry conversion:
0.26713, 0.30613, 0.33521, 0.55935.

The paper's raster-read K frequencies are 0.2657, 0.3136 and 0.3576, with
gaps of 0.045 (gap21) and 0.044 (gap32). The MPB gaps are about 0.039 and
0.029, so **the absolute frequency scale does not match**.

## Verdicts

- E7I5E: `DAI_FR00_PUBLIC_SOURCE_REFERENCE_LIMITATION_FORMALLY_CLOSED`.
  - Polarisation was `DISFAVORED_AS_PRIMARY_ROOT_CAUSE`.
  - Material semantics are a `POSSIBLE_PARTIAL_FREQUENCY_SCALE_CONTRIBUTOR_NOT_SOURCE_BOUND`.
  - Remaining ambiguity: `PUBLIC_ARTIFACTS_INSUFFICIENT_FOR_EXACT_COMSOL_CONTRACT`.
  - Composite Chern, individual Chern and local Berry reproduction were all
    `NOT_ESTABLISHED`. The decision was `no_further_dai_parameter_hunt`.
- E9C: `RANK1_K_KPRIME_BERRY_EVIDENCE_COMPLETE_WITH_E4C_REVIEW_REQUIRED`.
  The paper targets for Ω/a² at K′ are −0.92, +0.72 and +0.19 for bands 0–2.
- E9D/E9E: the FR0 and FR0.4 local Berry distributions compared with the
  paper stayed largely `UNQUALIFIED` (for example `paper_1_36_band2/3`).

## For LegumePhC

This is a cheap first Legume-vs-MPB benchmark. The geometry is one polygon
motif on a triangular lattice. Compare Γ/K/M frequencies with the R96 column
at increasing `gmax`. Do not re-open the paper parameter hunt.

## Provenance

- `MPW:audit/e7i5d/`, `e7i5e/closure.json`
- `MPW:audit/e9a/result.json`, `e9b/result.json`, `e9c/result.json`,
  `e9d/`, `e9e/`
