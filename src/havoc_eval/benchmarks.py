from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkItem:
    name: str
    prompt: str
    expected: str


def default_benchmarks() -> List[BenchmarkItem]:
    return [
        BenchmarkItem(
            name="anova_smoke",
            prompt="Perform an ANOVA on a two-factor experiment and report p-value",
            expected="ANOVA",
        ),
        BenchmarkItem(
            name="doe_design",
            prompt="DESIGN_DOE: Box-Behnken with factors Temp, Speed, Pressure",
            expected="DOE",
        ),
    ]
