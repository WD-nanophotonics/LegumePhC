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
```

Every run creates a new directory under `results/`; existing results are never
overwritten.  A run stores `config.json`, `summary.json`, `fields.npz`, and a
small diagnostic figure.

