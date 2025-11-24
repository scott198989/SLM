from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Argument:
    direction: str
    evidence: List[str]
    confidence: float
    text: str


def build_arguments(prompt: str) -> List[Argument]:
    pro = Argument(direction="PRO", evidence=["tool outputs"], confidence=0.6, text="Pro position placeholder")
    con = Argument(direction="CON", evidence=["counter-points"], confidence=0.4, text="Con position placeholder")
    return [pro, con]
