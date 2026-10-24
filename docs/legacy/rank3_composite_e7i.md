# Rank-3 composite manifold and composite valley Chern (MePhC E7I2–E7I4)

Geometry family: TriLatt triangular holes, fill-ratio endpoints FR00 and
FR050, MPB resolutions 32/48/64. Plaquettes were R48 dk=1/36 and 1/72, and
R64 dk=1/36.

## E7I2: the rank-2 pair is blocked, the rank-3 manifold qualifies

`E7I2_CLOSED_RANK3_QUALIFIED_TARGET_RANK2_PHYSICALLY_BLOCKED`

- The target rank-2 pair, bands [1,2] (0-based), fails at FR050 with
  `PLAQUETTE_REFINEMENT_UNQUALIFIED`.
- Interpretation: low-three-band manifold mixing (insufficient rank-2
  isolation) is supported. An H-space representation defect is not.
- Gates used:
  - min singular value 0.9
  - max principal angle 0.45
  - max projector distance 0.3
  - min external gap 0.05

## E7I3: rank-3 Wilson algebra

`RANK3_H_TO_EH_REPRESENTATION_BRIDGE_READY_FOR_SUPERVISOR_AUDIT`

- EH rank-3 spectral isolation and Wilson algebra: `QUALIFIED`.
- FR00 small-loop scaling: `SUPPORTED_WITH_VISIBLE_NUMERICAL_SENSITIVITY`.
  The density proxy is about −0.0090 at R48 and −0.0095 at R64, with the same
  sign; the resolution envelope is about 5%.
- FR050: `BOTH_SYMMETRY_SUPPRESSED`.
- `PHYSICAL_MAXWELL_BERRY_INTERPRETATION = NOT_YET_AUTHORIZED`. Chern numbers,
  Hall response and the rank-2 doublet observable were all *not authorized*.

## E7I4: first composite valley Chern (FR00)

`FIRST_FR00_COMPOSITE_VALLEY_CHERN_OBTAINED`

| Quantity | Value |
|---|---:|
| Stage-1 composite rank-3 valley Chern | **3.917e-4** |
| Stage-1 curvature integral | 2.461e-3 |
| Stage-2 (refined grid, E7I4F) | 3.734e-4 |
| Qualified elements | 331, covering the full area |
| Low-gap blockers recovered | 2 centres + 19 recovered, all passed |
| Min relative gap g34 | 0.048 (absolute 0.033) |

- Grid-refinement area deltas were 0.405 at 1/36, 0.0064 at 1/72, 0.0053 at
  1/144 and 0 at 1/288.
- Per-band, rank-2 and full-BZ Chern numbers were `NOT_AUTHORIZED`.

**Reading:** the composite rank-3 valley Chern for FR00 is effectively zero
(about 4e-4). This matches the expectation that the three-band composite is
topologically trivial when the gap above it is open.

## For LegumePhC

`solve_berry(rank=3)` already gives composite Wilson loops. The next step is
the valley-domain integral (see `valley_chern_e9f.md`), checked on FR00
against 3.9e-4.

## Provenance

- `MPW:audit/e7i2g/closure.json`
- `MPW:audit/e7i3a/`, `MPW:audit/e7i3b/`, `MPW:audit/e7i3c/result.json`
- `MPW:audit/e7i4b/`, `e7i4c/`, `e7i4d/`, `e7i4e/result.json`, `e7i4f/`
