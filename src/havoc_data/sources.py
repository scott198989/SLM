from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DataSource:
    name: str
    paths: List[str]
    weight: float = 1.0

    def files(self) -> List[Path]:
        collected = []
        for path in self.paths:
            p = Path(path)
            if p.is_dir():
                collected.extend(p.rglob("*.txt"))
            elif p.is_file():
                collected.append(p)
        return collected
