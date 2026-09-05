"""Verify the pinned Windows Python environment against requirements.lock."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def main() -> int:
    lock = ROOT / "requirements.lock"
    requirements = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            requirements[name.lower()] = version
    installed = {}
    missing = {}
    mismatch = {}
    for name, expected in requirements.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing[name] = expected
        else:
            installed[name] = actual
            if actual != expected:
                mismatch[name] = {"expected": expected, "actual": actual}
    report = {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()), "python_ok": sys.version_info[:2] == (3, 12), "lock": str(lock), "missing": missing, "mismatch": mismatch, "status": "PASS" if sys.version_info[:2] == (3, 12) and not missing and not mismatch else "FAIL"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
