from __future__ import annotations

from dataclasses import dataclass
from typing import List

from havoc_srs.arbiter import ArbiterDecision


@dataclass
class AuditReport:
    issues: List[str]
    severity: str
    downgraded_confidence: float


def run_audit(decision: ArbiterDecision, arguments) -> AuditReport:
    issues: List[str] = []
    confidence = decision.confidence
    if decision.winner == "UNDECIDED":
        issues.append("No clear argument selected")
        confidence *= 0.5
    if confidence < 0.5:
        issues.append("Low confidence")
    severity = "HIGH" if issues else "LOW"
    return AuditReport(issues=issues, severity=severity, downgraded_confidence=confidence)
