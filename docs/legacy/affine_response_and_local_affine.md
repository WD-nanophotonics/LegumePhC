# Affine response, local affine phase space, and affine architecture (MePhC E8, E10, P-series, R1–R3)

## E8A: analytic weighted Berry-gradient identities (all PASSED)

These checks covered the observable `WEIGHTED_BERRY_CURVATURE_GRADIENT_FUNCTIONAL`.
All of the following passed:
- the affine reciprocal transform B = F^{-T}
- reciprocal determinant scaling
- pointwise covariance of the Berry two-form
- Chern-measure invariance: −2.934568 in both q and k coordinates, error 0
- coordinate reparametrisation invariance
- finite-domain integration-by-parts boundary accounting: the direct and IBP
  dipoles agree to 3e-17

Other findings:
- A basis change alone produces zero physical response, within 6e-17.
- The full-BZ unweighted derivative is zero by periodicity.
- The unweighted valley derivative is `BOUNDARY_DEPENDENT_NOT_BULK_INVARIANT`.
- Physical status: `GEOMETRIC_FUNCTIONAL_VALIDATED_DYNAMICAL_OBSERVABLE_NOT_YET_DERIVED`.

**LegumePhC mapping:** these identities are exact unit-test targets for
`src/legumephc/dipole.py` (`compute_berry_dipole`), and they are not yet
ported.

## E8B: first live affine-weighted response (MPB)

Verdict: `E8B_FIRST_LIVE_AFFINE_WEIGHTED_RESPONSE_FULLY_AUDITABLE_AND_READY_FOR_FINAL_SUPERVISOR_SEAL`.

- All Gauss–Hermite nodes qualified: 27/27 at gh3 and 75/75 at gh5.
- Max projector distance 0.0056; min singular value 0.99999.
- Max relative stencil difference 8.0e-4.
- First response component: `SMALL_AND_QUADRATURE_SENSITIVE`.

## Local affine deformation (P1–P107, E10)

Model: uniform-strain family F(s) = diag(eˢ, e⁻ˢ) on a mirror-symmetric
lattice. The mixed phase-space curvatures Ω_{q,s} were treated as dimensionless
canonical quantities.

| Quantity | Value | Status |
|---|---:|---|
| Ω_{qx,s} | −0.19127 | symmetry-allowed, solver-derived |
| Ω_{qy,s} at q_x = 0 | 0 exactly | **forced zero by mirror plus time-reversal** (`omega_qy_s_symmetry_forced_zero`) |
| ∂ω/∂s | 0.0090296 | |
| Ω_{qx,qy} (rank-1, band 0) | −12.1287 (primary), −12.1289 (refined) | relative refinement 1.3e-5 |
| ∂ω/∂q_y | −0.093469 | |

- The nonzero Ω_{qy,s} values seen in P91–P100 were classified as
  *discrete-solver symmetry-breaking residuals*, not physics. This is the same
  lesson as the C3 work.
- Dimensionless trajectory benchmark:
  - Classification: `CANONICAL_DIMENSIONLESS_VALIDATION_ONLY_NOT_A_PHYSICAL_DEVICE_PREDICTION`.
  - With a deformation gradient of 1e-3 along x and τ = 0.5, the transverse
    displacement is −0.046726.
  - Analytic-vs-numeric residual: 4e-17.
- E10A gates for any future physical claim:
  - local-Bloch limit: ε_ad = a/L_def ≤ 0.05 and a·max|∇s| ≤ 0.05
  - minimum signed shift: |Δr⊥| ≥ max(0.1a, 3δ_sample)
  - uncertainty: U ≤ 0.5|Δr⊥|
  - Status: `THEORY_OR_FRAMEWORK_EXTENSION_REQUIRED_BEFORE_LIVE_PILOT`.

## Affine architecture specs R1–R3 (design, not results)

These non-negotiable semantics carry straight into LegumePhC's `Model2D`:

- The canonical affine map is an immutable 2×2 matrix. Singular or
  near-singular maps are rejected, and the composition order is explicit.
- The deformed BZ is built by **half-plane clipping against reciprocal-lattice
  points**, never by applying F^{-T} to the old hexagon. Its area must equal the
  reciprocal primitive-cell area.
- A **reference family is not a current symmetry.** A deformed triangular
  lattice must not claim C3, use HBZ/minimal-domain sampling, or present the
  old Γ/K/M as current high-symmetry points.
- The reciprocal convention omits 2π, so Aᵀ B = I. 2π enters at a single
  conversion point.

LegumePhC already implements these:
- `geometry.Affine2D` and `Lattice2D`
- `first_bz_vertices`, with 2π omitted
- `solver.q_to_legume_k` as the single 2π entry point
- `Model2D.point_group`, set only for the identity affine map

## Provenance

- `MPW:audit/e8a/result.json`, `e8b/closure.json`, `e10a/`–`e10f/`
- `MPW:audit/local_affine/p101_local_phase_space_symmetry_closure.json`
- `WSL:.local/share/mephc-runtime/science/results/` (schemas
  `mephc-local-affine-*`: symmetry closure, trajectory coefficient extraction,
  dimensionless trajectory benchmark)
- `WSL:AIInbox/new/MEPHC/*/attachments/MePhC_Affine_Architecture_R1*_Task.md`, `R2`, `R3`
