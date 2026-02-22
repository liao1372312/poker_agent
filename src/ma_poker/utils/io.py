from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict, got {type(data)} from {p}")
    return data


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def mkdirs(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

