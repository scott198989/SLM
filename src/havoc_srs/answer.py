from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from havoc_srs.audit import AuditReport
from havoc_srs.arbiter import ArbiterDecision
from havoc_srs.execute import ExecutionResult
from havoc_srs.ground import GroundedContext


@dataclass
class Answer:
    conclusion: str
    key_numbers: Dict[str, Any]
    assumptions: List[str]
    confidence: float
    caveats: List[str]
    suggested_checks: List[str]


def build_answer(
    prompt: str,
    decision: ArbiterDecision,
    audit: AuditReport,
    exec_result: ExecutionResult,
    grounded: GroundedContext,
) -> Answer:
    conclusion = f"Decision: {decision.winner}"
    key_numbers = {"confidence": audit.downgraded_confidence}
    assumptions = ["Placeholder assumptions"]
    caveats = audit.issues
    suggested_checks = ["Review DOE design", "Validate SPC data"]
    return Answer(
        conclusion=conclusion,
        key_numbers=key_numbers,
        assumptions=assumptions,
        confidence=audit.downgraded_confidence,
        caveats=caveats,
        suggested_checks=suggested_checks,
    )
