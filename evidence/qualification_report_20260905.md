# Public-evidence qualification report — 2026-09-05

Scope is restricted to evidence reproducible from this public LegumePhC
repository and its committed Legume runs. No local MePhC checkout, private
artifact, or unpublished MPB output was copied or used. Consequently, any
MPB comparison requested below is explicitly marked unqualified when no
public compact MPB record was available.

## 1. Are the frequencies C3?

For the default Legume reciprocal truncation, not yet qualified: the actual
gvec set is not closed under the exact affine map
`R(q+G)=q_next+G_next`, including `q_next-Rq=R*K-K`. At gmax 7, the maximum
orbit frequency residual is `3.865e-4` (G15) and `4.252e-4` (Circle). These
values decrease with cutoff, but the default finite basis is not an exact C3
representation.

The standalone closed-basis adapter is a causal control, not a replacement
for the default cutoff result. At seed gmax 5 it expands 81 vectors to 123
vectors and gives frequency residual `4.09e-10` (G15) and `2.82e-14`
(Circle), while preserving Legume's Fourier-matrix construction in local
project code only.

## 2. Are E/H energy scalars and rank-2 projectors C3?

The available Legume scalar is TE `H_z` density. The gauge-invariant composite
`[2,3]` density at gmax 7 has rotated residual `6.709e-3` (G15) and
`6.938e-3` (Circle), identical at grid sizes 32 and 64. A matching MPB E/H
energy-scalar comparison is not publicly evidenced and is therefore
unqualified.

Default-cutoff rank-2 projector covariance is also not C3-qualified: residuals
are `1.201e-2` (G15) and `1.232e-2` (Circle) at gmax 7. The local rank-2 Wilson
loops separately pass numerical gap/link/branch gates. In the closed-basis
adapter control, rank-2 projector residuals fall to `3.92e-9` (G15) and
`4.00e-14` (Circle), with TE operator covariance residuals `9.55e-9` and
`9.96e-16`.

## 3. At which layer does Berry first break?

No public MPB Berry orbit is available, so an MPB layer-by-layer answer is
unqualified. In the Legume evidence, the first exact obstruction is the
reciprocal-basis layer: the default gvec truncation misses 25.6% of mapped
vectors at gmax 6. After that, default-basis projector covariance remains
non-qualified. The local rank-2 plaquette Wilson calculations are numerically
qualified, with finest-step pairwise phase-density residual `5.884e-3`
(G15) and `6.457e-3` (Circle) at gmax 7. Rank-1 Berry is withheld because the
band gap gate fails.

## 4. What is the G16 versus G15 difference?

At the common, public Legume gmax 5 pilot, the maximum absolute frequency
difference between G16 and G15 over the three orbit points and four bands is
`9.47e-7`; their maximum frequency C3 residuals are `9.774e-4` and
`9.777e-4`, and rank-2 projector residuals are `2.11418e-2` and
`2.11452e-2`. The geometry contribution is therefore about three orders of
magnitude below the gmax 4→5 cutoff change (`1.051e-3`): approximately
`9.0e-4` of that cutoff contribution on this diagnostic.

G16 remains an approximate-C3 control by construction (`geometry residual
2.616e-2`), while G15 is strict (`2.118e-16`). No G16 gmax7 run was made,
as required by the bounded extension plan.

## 5. Is Legume sufficient to replace MPB?

Not qualified as a blanket replacement. Legume is sufficient for this
Windows-native 2-D PWE method and for the finite-cutoff causal diagnosis: its
homogeneous calibration is at `8.33e-17` maximum frequency error, its default
cutoff residuals co-converge with the missing-basis fraction, and the closed
basis adapter recovers numerical C3 covariance. It is not sufficient to claim
MPB-equivalent E/H scalar or Berry conclusions without a publicly verifiable
MPB cross-check under the same geometry, bands, and gauge-invariant metrics.

The earlier global three-member triangle Wilson path is retained only as
immutable history and is not local Berry evidence. All corrected results use
four-corner local plaquettes, separate numerical and C3 qualification, and
`RANK1_WITHHELD` whenever the rank-1 gate fails.

Evidence records:

- `results/20260905T124117.824695Z-pwe_reciprocal_c3_operator_diagnosis-6b1ecdcc/`
- `results/20260905T124226.982994Z-pwe_c3_gmax7-471c19a5/`
- `results/20260905T125441.544713Z-pwe_c3_closed_basis_adapter-9a91aab8/`
