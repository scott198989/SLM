"""
Global Verification for HAVOC PRIME

Checks reasoning for contradictions and quality issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from havoc_prime.operator_graph import OperatorGraph
from havoc_prime.workspace import GlobalWorkspace


@dataclass
class VerificationReport:
    """Report from global verification"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_penalty: float = 0.0


class GlobalVerification:
    """
    Global verification and consistency checking.

    Checks for:
    - Contradictions between subgoals
    - Violated constraints
    - Inconsistent facts
    - Logic errors
    """

    def verify(
        self,
        workspace: GlobalWorkspace,
        operator_graph: OperatorGraph
    ) -> VerificationReport:
        """
        Run global verification checks.

        Args:
            workspace: Global workspace
            operator_graph: Operator graph with subgoals

        Returns:
            VerificationReport with issues and warnings
        """
        report = VerificationReport(passed=True)

        # Check 1: Constraint violations
        violated = workspace.get_violated_constraints()
        if violated:
            for constraint in violated:
                if constraint.severity in {"HIGH", "CRITICAL"}:
                    report.issues.append(f"Constraint violated: {constraint.description}")
                    report.confidence_penalty += 0.1
                    if constraint.severity == "CRITICAL":
                        report.passed = False
                else:
                    report.warnings.append(f"Minor constraint violated: {constraint.description}")

        # Check 2: Failed subgoals
        failed_subgoals = [sg for sg in operator_graph.subgoals if sg.status == "failed"]
        if failed_subgoals:
            report.issues.append(f"{len(failed_subgoals)} subgoal(s) failed")
            report.confidence_penalty += 0.05 * len(failed_subgoals)

        # Check 3: Low confidence facts
        low_conf_facts = workspace.get_low_confidence_facts(threshold=0.3)
        if low_conf_facts:
            report.warnings.append(f"{len(low_conf_facts)} fact(s) with very low confidence")
            report.confidence_penalty += 0.05

        # Check 4: Critical assumptions
        critical_assumptions = workspace.get_critical_assumptions()
        if critical_assumptions:
            report.warnings.append(f"Relies on {len(critical_assumptions)} critical assumptions")

        # Check 5: Contradictions (simplified check)
        contradictions = self._check_contradictions(workspace)
        if contradictions:
            report.issues.extend(contradictions)
            report.passed = False
            report.confidence_penalty += 0.2

        # Cap penalty
        report.confidence_penalty = min(0.5, report.confidence_penalty)

        return report

    def _check_contradictions(self, workspace: GlobalWorkspace) -> List[str]:
        """Check for contradictory facts"""
        contradictions = []

        # Example: Check if both "in_control=True" and "in_control=False" exist
        facts = workspace.get_all_facts()

        # Simple contradiction detection (expandable)
        if "in_control" in facts and "control_violations" in facts:
            if facts["in_control"] and facts["control_violations"] > 0:
                contradictions.append("Contradiction: Process marked in-control but violations detected")

        if "significant" in facts and "pvalue" in facts:
            if facts["significant"] and facts["pvalue"] > 0.05:
                contradictions.append("Contradiction: Marked significant but p-value > 0.05")

        return contradictions
