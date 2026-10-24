# Legacy research carried over from MePhC

These notes preserve the scientific conclusions of the retired MePhC work
(MPB/Meep in WSL, plus the `MePhC-Windows` clone and the `MePhC-TriLatt` /
`MePhC-SqrLatt` workspaces). They are **written summaries with small numeric
tables only**. No raw legacy dataset, record, or pickle is copied into this
repository.

Rules for reading them:

- Every status label is quoted from the legacy record as it was. A summary
  never upgrades a result: `UNQUALIFIED`, `NOT_ESTABLISHED`, `FAIL_CLOSED` and
  `NOT_AUTHORIZED` stay what they were.
- MPB numbers are a different discretisation (real-space grid, resolution
  R) from LegumePhC's plane-wave cutoff `gmax`. They are comparison targets,
  never LegumePhC evidence.
- Provenance paths refer to the legacy machine as of 2026-10-24:
  - `WSL:` = `\\wsl.localhost\Ubuntu\home\icy`
  - `MPW:` = `C:\Users\icywo\PycharmProjects\MePhC-Windows` (git
    `WD-nanophotonics/MePhC`, branch `sandbox` @ `cf01ed7`)
  - the same `audit/` tree is in every MePhC checkout and on GitHub.

## Index

| Note | Topic |
|---|---|
| [mpb_c3_defect.md](mpb_c3_defect.md) | Why the MPB triangular-lattice Berry map was not C3-consistent (M1–M67) |
| [valley_chern_e9f.md](valley_chern_e9f.md) | Half-BZ valley Chern contract and the only qualified values |
| [rank3_composite_e7i.md](rank3_composite_e7i.md) | Rank-3 composite manifold, Wilson algebra, first composite valley Chern |
| [dai_paper_reproduction.md](dai_paper_reproduction.md) | Attempted reproduction of the Dai triangular-hole paper (E7I5, E9A–E9E) |
| [affine_response_and_local_affine.md](affine_response_and_local_affine.md) | Weighted Berry gradient, affine response, local affine phase-space work, affine architecture specs |
| [trilatt_mpb_reference.md](trilatt_mpb_reference.md) | MPB band-frequency reference table for TriLatt geometries |

## Is it already in LegumePhC?

| Legacy finding | LegumePhC status |
|---|---|
| G15/G16 strict vs. approximate C3 controls, M7 orbit around K=(2/3,0) | Inherited: `configs/c3_benchmark.json`, `evidence/pilot_20260905.md` |
| Rank-1 withholding; numerical vs. C3 qualification kept separate | Inherited: `src/legumephc/berry.py`, `evidence/corrected_pilot_20260905.md` |
| C3 breaking is a solver-discretisation artefact, not geometry | Inherited in PWE form: non-closed cutoff fixed by orbit-closed basis (`evidence/causal_diagnosis_20260905.md`, `evidence/closed_basis_berry_20260905.md`). The MPB-specific chain is recorded only here. |
| G16-vs-G15 geometry effect much smaller than truncation effect | Inherited: `evidence/qualification_report_20260905.md` §4 |
| Local-plaquette vs. global-triangle Wilson correction | Inherited: `evidence/corrected_pilot_20260905.md` |
| Generic affine models must not inherit C3/C4; BZ by half-plane clipping | Inherited: `Model2D.point_group` (identity-only), `geometry.first_bz_vertices` |
| Weighted Berry-curvature gradient / dipole functional | Partly: `src/legumephc/dipole.py`; analytic E8A identities not yet ported as tests |
| Half-BZ valley Chern number | **Open** (no LegumePhC operation yet) |
| Rank-3 composite Wilson / composite valley Chern | **Open** (`solve_berry` supports composite rank, no valley integration) |
| Dai paper benchmark | **Open**; reference numbers below |
| Local non-uniform affine, mixed phase-space curvature, wavepacket shift | **Open**; contract and gates below |
| MPB TriLatt band reference | **Open** comparison target |

## Open science leads for LegumePhC

1. **Valley Chern over the K half-BZ** with Legume. Use the orbit-closed basis,
   then compare with the legacy band0 −0.094 / band1 +0.509 values (MPB, R128,
   threshold 0.02).
2. **Rank-3 composite valley Chern** for the FR00 triangular-hole geometry
   (legacy stage-1 value 3.9e-4).
3. **Affine response**: port the E8A analytic identities as unit tests of
   `compute_berry_dipole`, then repeat the E8B affine-weighted response with
   Legume.
4. **Local affine phase space**: the mixed curvature Ω_{qx,s} for
   F(s)=diag(eˢ,e⁻ˢ), and the dimensionless trajectory benchmark.
5. **Legume vs. MPB band check** on the TriLatt geometries.
