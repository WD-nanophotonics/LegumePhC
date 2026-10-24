"""Convert NumPy-bearing Studio payloads into plain JSON values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_json(value: Any) -> Any:
    """Return a JSON-safe copy; complex arrays are refused, never silently cast.

    Raw complex eigenvectors are not a scientific observable in this project,
    so they are dropped by :func:`real_arrays` before reaching this function.
    """

    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            raise TypeError("complex arrays are not exported to the Studio client")
        return to_json(value.tolist())
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def real_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Keep only real-valued arrays for display."""

    return {name: to_json(array) for name, array in arrays.items() if not np.iscomplexobj(array)}
