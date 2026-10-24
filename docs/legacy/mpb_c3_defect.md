# MPB C3 defect in the triangular-lattice Berry map (MePhC M1–M67)

Goal ID `MEPHC-BERRY-C3-CONSISTENCY-V1`, 2026-09-03 → 2026-09-05.

## Question

A TriLatt Berry-curvature map was not C3-symmetric even though the geometry
nominally was. The frozen observation was band 2, r1=80 nm / r2=75 nm 16-gon
holes, a=400 nm, n_eff=2.7, resolution 128, step 0.001, a 36-point K-centred
half-BZ grid. Its worst C3 orbit (the M7 orbit, K−(7/36,0) and its 120°/240°
images about K=(2/3,0)) had values about **−257.5, +194.0, +46.9**.

Controls: **G16** is the frozen geometry (not exactly C3). **G15** uses
area-matched 15-gons, an exact C3 control, with r15 = r16·1.0017919605440295
(r1=80.143357, r2=75.134397).

## Causal chain (verdicts quoted verbatim)

| Step | Verdict | Meaning |
|---|---|---|
| M2–M13 | `STOP_C3_GOAL_AS_MODEL_CONTRADICTION` (M13) | Rank-2 covariance could not be closed from stored states; band-window leakage ruled out as sole cause |
| M14–M21 | `STOP_C3_GOAL_AT_INSTALLED_MPB_INTROSPECTION_LIMIT` | The FFT/material operator could not be inspected through the installed MPB bindings |
| M22–M31 | H-field sampling and Fourier alias audits; stopped at the raw-field introspection limit | |
| M32–M38 | MPB 1.12.0 Build5 source recovered; `STOP_C3_GOAL_AS_VALIDATED_NATIVE_H_NUMERICAL_OR_STATE_FAMILY_CONTRADICTION` | |
| M39–M44 | Deterministic repeats and resolution plateaus on G15 | Not initialisation noise |
| M45–M48 | `FULL_NONASYMPTOTIC_RESOLUTION_REGIME`, `R256_MIXED_SEMANTIC_FAMILY` | Even R=256 is not in the asymptotic regime for C3 |
| M49R1 | `R256_MULTIPLE_C3_BREAKING_LAYERS` | Breaking appears at the frequency, gap and subspace layers |
| M50 | `R256_FREQUENCY_C3_BREAKING_FULLY_MESH_SENSITIVE` | Mesh 1 vs. mesh 3 changes all 24 identities |
| M51 | `R256_M3_M5_NO_FREQUENCY_MESH_PLATEAU_MESH_INSENSITIVE_C3_FAILURE` | Mesh 3 vs. 5 agree, but C3 still fails, so the defect is not the subpixel mesh |
| M52R3/M53R2 | `R256_M5_RECIPROCAL_WINDOW_STRUCTURAL_ONLY_NOT_CAUSAL` | MPB's reciprocal window is not C3-covariant, but this is not the cause. The earliest broken layer is L1 (frequency). |
| M54R1 | `R256_BASE_GEOMETRY_RASTER_OR_GRID_C3_BREAKING` | The rasterised ε grid breaks C3. The scalar residual max is 6.29 and the tensor Frobenius residual is 1.49. A C3-projected grid would pass. |
| M55R1–M58 | Every attempt to feed a C3-exact ε (public input, scalar, HDF5) failed readback | `R256_HDF5_TRANSFER_NONLINEAR` |
| M59R1 | `R256_CANONICAL_PRIMITIVE_CONTINUOUS_C3_PASS_RUNTIME_MATERIAL_BREAKS` | The continuous geometry is exactly C3; MPB's runtime material is not |
| M60 | `R256_CANONICAL_PRIMITIVE_NO_FREQUENCY_RESTORATION` | Fixing the primitive does not restore C3 frequencies |
| **M61R1** | **`R256_HOMOGENEOUS_K_OPERATOR_C3_BREAKS`** | **With a uniform medium, MPB's k-dependent operator still breaks C3.** The analytic reference passes exactly. All 48 native comparisons fail, with residuals of 6–7e-5 between the IDENTITY member and the C3/C3² members. Example (band 1, v0): 0.2138982 vs. 0.2138364. |
| M62/M63 | `..._ANALYTIC_ASSIGNMENT_UNMATCHED`, `R256_HOMOGENEOUS_RAW_SAME_SHELL_VALUE_DEFORMATION_TOLERANCE_INSENSITIVE` | Same-shell plane-wave values are deformed regardless of tolerance |
| M64–M64R4R3 | `R256_NATIVE_HOMOGENEOUS_DEFECT_NOT_LOCALIZED_NO_PATCH` | The defect source in MPB could not be localised; no patch was applied |
| M65 | `R256_M65_FREQUENCY_LAYER_PRECEDES_SCALAR_FIELD_LAYER` | |
| M66 | `R256_M66_FREQUENCY_C3_FAILURE_RECIPROCAL_PROJECTOR_SCALAR_BREAK` | C3 and C3² agree to about 1e-12; the unrotated point is off by about 6e-5 |
| M67R2 | `R256_M67R2_FREQUENCY_C3_FAILURE_TRUE_PROJECTOR_TRACE_BREAK` | Rank-2 projector distances are about 1.5 and Frobenius residuals 0.89–0.99 between orbit members |

## Conclusion

The MePhC C3 Berry asymmetry was a **solver-discretisation artefact of MPB**,
present even for a homogeneous medium, and not physics of the geometry. The
IDENTITY member of each orbit differs from the two rotated members, which
agree with each other. That pattern points to a k-coordinate-dependent
operator construction rather than geometry.

LegumePhC independently found the plane-wave analogue: a circular `gmax`
cutoff that is not orbit-closed. It fixes this with a verified C3/C4-closed
basis (`basis_policy=auto`); see `evidence/causal_diagnosis_20260905.md` and
`evidence/closed_basis_berry_20260905.md`. This is the main scientific reason
LegumePhC exists.

## Provenance

- `WSL:.local/share/mephc-runtime/science/results/MEPHC-SCIENCE-*.json`
  (180 run verdicts, the `mephc-berry-c3-consistency-m*` schemas). Kept in
  place during the cleanup.
- `MPW:audit/berry_c3_consistency/PLAN.md`, `m*_*.py`
- `MPW:vendor/mpb_c3_patch/README.md`
- Frozen observation: `WSL:TriLatt/data/_tmp/bc_latest.pkl`, SHA-256
  `d602a094…a88d`
