from __future__ import annotations

from dataclasses import dataclass
from typing import List

from havoc_srs.argue import Argument


@dataclass
class ArbiterDecision:
    winner: str
    confidence: float
    rationale: List[str]


def decide(arguments: List[Argument]) -> ArbiterDecision:
    if not arguments:
        return ArbiterDecision(winner="UNDECIDED", confidence=0.0, rationale=["No arguments"])
    winner = max(arguments, key=lambda a: a.confidence)
    return ArbiterDecision(winner=winner.direction, confidence=winner.confidence, rationale=[winner.text])
