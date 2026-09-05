# LegumePhC

Canonical Windows-native, pure-Python 2D platform using
[Legume](https://legume.readthedocs.io/). MePhC/MPB/WSL are legacy read-only
references only; this repository deliberately contains no Thin Flow, Courier,
Meep, MPB, WSL, or dataset ledger.

## PyCharm / Windows setup

Select `.venv\\Scripts\\python.exe` as the project interpreter.  Then run
`run_c3_crosscheck.py` directly.  The default is a cheap homogeneous-medium
calibration; `--smoke` runs the frozen three-point G15 Legume PWE pilot at
`gmax=2`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe run_c3_crosscheck.py
.venv\Scripts\python.exe run_c3_crosscheck.py --smoke --case G15 --gmax 2
.venv\Scripts\python.exe run_c3_crosscheck.py --convergence
.venv\Scripts\python.exe run_c3_crosscheck.py --extension
.venv\Scripts\python.exe run_c3_crosscheck.py --operator-diagnosis
.venv\Scripts\python.exe run_c3_crosscheck.py --closed-adapter
.venv\Scripts\python.exe run_c3_crosscheck.py --closed-plaquette
.venv\Scripts\python.exe run_c3_crosscheck.py --bandpath
.venv\Scripts\python.exe run_c3_crosscheck.py --berry-map
```

`--convergence` runs the three-geometry `gmax=2,3,4,5` pilot with local
four-corner plaquettes, density/projector covariance, and Wilson diagnostics.
`--extension` runs the strict G15 and Circle controls through `gmax=6`.
`--operator-diagnosis` quantifies affine-C3 reciprocal-basis closure, and
`--closed-adapter` tests a standalone C3-closed Fourier basis without editing
Legume's installed package.
`--closed-plaquette` performs the local Wilson and TE H/E/energy scalar
validation on that closed basis.
`--bandpath` compares the G15 Γ–K–M–Γ path, and `--berry-map` samples an
independent 5×5 G15 rank-2 map with raw and C3-residual outputs.

Every run creates a new directory under `results/`; existing results are never
overwritten. A new run stores only `config.json` (including identity),
`summary.json`, `arrays.npz`, and a `figures/` directory.

The stable domain surface is `Model2D`, `solve_bands`, and `frequency_at_k`.
`basis_policy="auto"` uses only verified finite C3/C4 closures; native or
circular truncation is available as an explicit comparison control. The PWE
eigensolver uses a documented `+1` numerical shift before subtracting it from
the squared eigenvalues.

Phase B adds `compute_field_observables` (gauge-invariant `E2`, `H2`, energy,
and region ratios), `solve_efs` (frequency samples ready for iso-frequency
contours), `solve_berry` (rank-1/composite Wilson qualification with raw
unsymmetrized output), and `compute_berry_dipole` (geometric gradient and
first moment). Dipole output becomes a physical response only when an
explicit frequency window and occupation/response weight are supplied.

## Phase C studies

From PyCharm, select `.venv\Scripts\python.exe` and run any study script
directly. The scripts do not solve on import; each run writes fresh immutable
records below its own ignored `studies/<name>/results/` directory.

```powershell
.venv\Scripts\python.exe studies\triangular\study.py
.venv\Scripts\python.exe studies\square\study.py
.venv\Scripts\python.exe studies\affine\study.py
```

`triangular` freezes G15/Circle/G16 and reports verified C3 versus near-C3
behavior, including separate Berry gate and convergence status. The single
`gmax=2` Berry smoke is intentionally convergence-unassessed. `square` uses
the verified one-circle C4 model and Γ–X–M–Γ path.
`affine` applies the configured full 2×2 transform, ellipse/material-mask
geometry, TE/TM solves, and reports symmetry `none`. The read-only
`references/mpb/index.json` file is metadata only and is not read by studies.

## Phase D Studio

Launch the compact Tkinter Studio directly with the pinned Windows interpreter:

```powershell
.venv\Scripts\python.exe studio.py
```

Projects use the `.legumephc-studio.json` format and store only relative,
hashed references to immutable JSON+NPZ records. Parameter presets use the
separate `.legumephc-preset.json` format and never contain result references.
Preview refreshes are solver-free; formal calculations run in one exact child
process and can be cancelled without a background service. User-facing band
labels are one-based and are mapped to the core's explicit zero-based API.

## Phase E acceptance

Verify the pinned Windows environment and run the finite 18-operation acceptance
matrix in one foreground command:

```powershell
.\phase_e_acceptance.cmd --cycles 1 --gmax 2
```

Each invocation creates a fresh ignored `acceptance/<run-id>/` directory with
per-operation timing, result identities, immutable records, and a machine-
generated `manifest.json`; it never resumes or overwrites an earlier run. The
short matrix covers triangular, square, and affine preview/frequency/band/EFS/
field/Berry paths and rejects invalid qualification labels. The prepared
unattended command is:

```powershell
.\phase_e_acceptance.cmd --duration-hours 6 --cycles 6 --gmax 2
```

It is a single foreground command expected to run for at least six hours. It
was prepared but not started during acceptance.
