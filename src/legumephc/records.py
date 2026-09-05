from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_run(root: str | Path, label: str, config: dict, summary: dict, arrays: dict[str, np.ndarray]) -> Path:
    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = parent / f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"
    target.mkdir()
    _json(target / "config.json", config)
    np.savez_compressed(target / "fields.npz", **arrays)
    _json(target / "summary.json", summary)
    return target

