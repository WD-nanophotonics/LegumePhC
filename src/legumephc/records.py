from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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
    canonical = json.dumps({"identity": identity, "config": config}, sort_keys=True, separators=(",", ":"))
    record_identity = {**identity, "cache_identity": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    _json(target / "config.json", {"identity": record_identity, "config": config})
    _json(target / "summary.json", summary)
    np.savez_compressed(target / "arrays.npz", **arrays)
    (target / "figures").mkdir()
    return target


def create_model_record(root, model, operation: str, config: dict, summary: dict, arrays: dict[str, np.ndarray]) -> Path:
    """Write one immutable public-operation record for a Model2D instance."""

    identity = {
        "model": "Model2D",
        "geometry": model.geometry.name,
        "affine": {"linear": model.affine.linear.tolist(), "translation": model.affine.translation.tolist()},
        "basis": model.identity["basis"],
        "solver": "Legume.PlaneWaveExp",
        "operation": operation,
    }
    config = {**config, "actual_lattice_constant_m": model.actual_lattice_constant_m}
    return create_record(root, identity=identity, config=config, summary=summary, arrays=arrays)
