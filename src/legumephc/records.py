from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np


IDENTITY_FIELDS = ("model", "geometry", "affine", "basis", "solver", "operation")


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_record(
    root: str | Path,
    *,
    identity: dict[str, object],
    config: dict,
    summary: dict,
    arrays: dict[str, np.ndarray],
) -> Path:
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"record identity is missing: {', '.join(missing)}")
    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    label = str(identity.get("operation", "run"))
    target = parent / f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"
    target.mkdir()
    _json(target / "config.json", {"identity": identity, "config": config})
    _json(target / "summary.json", summary)
    np.savez_compressed(target / "arrays.npz", **arrays)
    # Keep the legacy filename readable while all new records use arrays.npz.
    np.savez_compressed(target / "fields.npz", **arrays)
    (target / "figures").mkdir()
    return target


def create_run(root: str | Path, label: str, config: dict, summary: dict, arrays: dict[str, np.ndarray]) -> Path:
    """Backward-compatible wrapper for the immutable record core."""

    identity = {
        "model": summary.get("model", "Model2D"),
        "geometry": summary.get("case", summary.get("geometry", "unknown")),
        "affine": summary.get("affine", "identity"),
        "basis": summary.get("basis_policy", summary.get("basis", "native")),
        "solver": summary.get("solver", "Legume.PlaneWaveExp"),
        "operation": label,
    }
    return create_record(root, identity=identity, config=config, summary=summary, arrays=arrays)
