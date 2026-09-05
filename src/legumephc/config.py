from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    raw: dict[str, Any]

    @property
    def lattice_constant_nm(self) -> float:
        return float(self.raw["lattice_constant_nm"])

    @property
    def background_epsilon(self) -> float:
        return float(self.raw["background_epsilon"])

    @property
    def inclusion_epsilon(self) -> float:
        return float(self.raw["inclusion_epsilon"])


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "c3_benchmark.json"


def load_benchmark(path: str | Path | None = None) -> BenchmarkConfig:
    source = default_config_path() if path is None else Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema") != "legumephc-c3-benchmark-v1":
        raise ValueError("unsupported benchmark schema")
    if int(raw["target_band_one_based"]) != 2:
        raise ValueError("the frozen benchmark targets band 2")
    return BenchmarkConfig(raw)

