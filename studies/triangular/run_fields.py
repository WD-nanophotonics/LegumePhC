import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.triangular.case import CASE
from studies.triangular.parameters import FIELDS
from studies.direct_runner import run_entrypoint

def main() -> int:
    return run_entrypoint("triangular", CASE, FIELDS)

if __name__ == "__main__":
    raise SystemExit(main())
