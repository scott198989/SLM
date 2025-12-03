"""
Global Workspace for HAVOC PRIME

Shared memory for facts, constraints, assumptions, and partial results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Fact:
    """A fact stored in the workspace"""
    key: str
    value: Any
    confidence: float = 1.0
    source: str = "unknown"


@dataclass
class Assumption:
    """An assumption made during reasoning"""
    description: str
    confidence: float = 0.8
    critical: bool = False


@dataclass
class Constraint:
    """A constraint that must be satisfied"""
    description: str
    violated: bool = False
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL


class GlobalWorkspace:
    """
    Shared memory accessible across all subgoals.

    Think of this as the "global brain workspace" where:
    - Facts are stored and retrieved
    - Assumptions are tracked
    - Constraints are enforced
    - Partial results accumulate
    - Confidence is propagated

    All subgoals read from and write to this workspace.
    """

    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.assumptions: List[Assumption] = []
        self.constraints: List[Constraint] = []
        self.partial_results: Dict[str, Any] = {}
        self.subgoal_history: List[Dict[str, Any]] = []
        self.global_confidence: float = 1.0

    def add_fact(self, key: str, value: Any, confidence: float = 1.0, source: str = "unknown") -> None:
        """Add or update a fact"""
        self.facts[key] = Fact(
            key=key,
            value=value,
            confidence=confidence,
            source=source
        )

    def get_fact(self, key: str) -> Optional[Fact]:
        """Retrieve a fact"""
        return self.facts.get(key)

    def add_assumption(self, description: str, confidence: float = 0.8, critical: bool = False) -> None:
        """Record an assumption"""
        self.assumptions.append(Assumption(
            description=description,
            confidence=confidence,
            critical=critical
        ))

    def add_constraint(self, description: str, severity: str = "MEDIUM") -> None:
        """Add a constraint"""
        self.constraints.append(Constraint(
            description=description,
            violated=False,
            severity=severity
        ))

    def mark_constraint_violated(self, description: str) -> None:
        """Mark a constraint as violated"""
        for constraint in self.constraints:
            if constraint.description == description:
                constraint.violated = True
                self._downgrade_confidence_for_violation(constraint.severity)
                break

    def store_partial_result(self, subgoal_id: str, result: Any) -> None:
        """Store intermediate result from a subgoal"""
        self.partial_results[subgoal_id] = result

        # Record in history
        self.subgoal_history.append({
            "subgoal_id": subgoal_id,
            "result": result,
            "timestamp": len(self.subgoal_history)
        })

    def get_partial_result(self, subgoal_id: str) -> Optional[Any]:
        """Retrieve partial result"""
        return self.partial_results.get(subgoal_id)

    def update_global_confidence(self, new_confidence: float) -> None:
        """
        Update global confidence.

        Confidence can only decrease, never increase (pessimistic updates).
        """
        if new_confidence < self.global_confidence:
            self.global_confidence = new_confidence

    def get_violated_constraints(self) -> List[Constraint]:
        """Get all violated constraints"""
        return [c for c in self.constraints if c.violated]

    def get_critical_assumptions(self) -> List[Assumption]:
        """Get all critical assumptions"""
        return [a for a in self.assumptions if a.critical]

    def get_low_confidence_facts(self, threshold: float = 0.5) -> List[Fact]:
        """Get facts below confidence threshold"""
        return [f for f in self.facts.values() if f.confidence < threshold]

    def summarize(self) -> Dict[str, Any]:
        """
        Get workspace summary.

        Returns:
            Dict with counts and key metrics
        """
        return {
            "facts_count": len(self.facts),
            "assumptions_count": len(self.assumptions),
            "constraints_count": len(self.constraints),
            "violated_constraints_count": len(self.get_violated_constraints()),
            "partial_results_count": len(self.partial_results),
            "global_confidence": self.global_confidence,
            "critical_assumptions": len(self.get_critical_assumptions()),
            "low_confidence_facts": len(self.get_low_confidence_facts())
        }

    def get_all_facts(self) -> Dict[str, Any]:
        """Get all facts as dict (for inspection)"""
        return {k: v.value for k, v in self.facts.items()}

    def _downgrade_confidence_for_violation(self, severity: str) -> None:
        """Downgrade confidence based on constraint violation severity"""
        penalty_map = {
            "LOW": 0.95,
            "MEDIUM": 0.85,
            "HIGH": 0.70,
            "CRITICAL": 0.40
        }

        penalty = penalty_map.get(severity, 0.90)
        self.update_global_confidence(self.global_confidence * penalty)

    def clear(self) -> None:
        """Clear workspace (for testing or macro-loop restart)"""
        self.facts.clear()
        self.assumptions.clear()
        self.constraints.clear()
        self.partial_results.clear()
        self.subgoal_history.clear()
        self.global_confidence = 1.0
