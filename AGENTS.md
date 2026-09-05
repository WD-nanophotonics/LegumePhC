# LegumePhC migration route

LegumePhC is the Windows-native two-dimensional PWE primary platform. Work
directly in the repository `.venv` with CPython 3.12 and Legume 1.0.3. Never
invoke WSL, Meep, MPB, GME, Thin Flow, Courier, a service, or another worker.

The migration route is Phase A (trusted baseline and domain core) -> Phase B
(Studio/API consumers) -> only then any production migration. Science remains
the priority. The repository is a single `studies` workspace; old MPB data is
read-only reference and may be used only when it is already public and
explicitly identified. Do not copy local/private evidence into this repo.

Use local branch `sandbox` only and push only `origin/sandbox`; never move
`origin/main`. Preserve every scientific result directory. New immutable
records contain config, summary, arrays, figures, and an identity covering
model, geometry, affine, basis, solver, and operation. No database, ledger,
service, or workflow framework is needed.

`basis_policy=auto` is the production default. It may apply a verified finite
point-group orbit closure (C3/C4); circular/native truncation is retained only
as an explicit comparison control. Generic affine models must not be assigned
symmetry they do not have. Never compare raw complex eigenvectors for science:
use frequencies, scalar densities/energies, projectors, and gauge-invariant
Wilson loops. Rank-1 Berry is `RANK1_WITHHELD` whenever gap, association/link,
or branch qualification fails.

Keep the stable domain API small: `Model2D`, `solve_bands`, and
`frequency_at_k`. The legacy `run_c3_crosscheck.py` remains runnable while it
gradually delegates to that core; do not create a second solver implementation.
Native and custom-basis TE/TM paths must be compared on the same gvec before
any migration claim. The PWE zero-frequency numerical shift is documented in
the solver API and is not an unexplained production convention.

Before becoming idle, send the fixed supervisor task
`01a04136-7e60-75c3-88cf-156581a3733e` a compact handoff with Phase, source
SHA, evidence, tests, exact next action, and solver uncertainty. Do not create
or fork another worker. A milestone or negative result never terminates the
project; continue to the next safe scientific phase after handoff unless the
supervisor explicitly approves idle.
