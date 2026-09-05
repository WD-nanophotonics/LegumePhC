# LegumePhC

Windows-native, pure-Python cross-validation of the MePhC Berry/C3 question
using [Legume](https://legume.readthedocs.io/).  This repository deliberately
contains no Thin Flow, Courier, Meep, MPB, WSL, or dataset ledger.

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
```

`--convergence` runs the three-geometry `gmax=2,3,4,5` pilot with local
four-corner plaquettes, density/projector covariance, and Wilson diagnostics.
`--extension` runs the strict G15 and Circle controls through `gmax=6`.
`--operator-diagnosis` quantifies affine-C3 reciprocal-basis closure, and
`--closed-adapter` tests a standalone C3-closed Fourier basis without editing
Legume's installed package.
`--closed-plaquette` performs the local Wilson and TE H/E/energy scalar
validation on that closed basis.

Every run creates a new directory under `results/`; existing results are never
overwritten.  A run stores `config.json`, `summary.json`, `fields.npz`, and a
small diagnostic figure.
