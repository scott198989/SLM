"""
Final Compression for HAVOC PRIME

Compresses reasoning trace into clean, concise answer.
"""

from __future__ import annotations

from typing import Any, Dict

from havoc_prime.operator_graph import OperatorGraph
from havoc_prime.workspace import GlobalWorkspace


class FinalCompression:
    """
    Compresses full reasoning trace into clean answer.

    Takes verbose workspace state and produces:
    - Concise conclusion
    - Key numbers only
    - Essential caveats
    """

    def compress(
        self,
        workspace: GlobalWorkspace,
        operator_graph: OperatorGraph,
        synthesis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compress reasoning into final answer.

        Args:
            workspace: Global workspace
            operator_graph: Operator graph
            synthesis_result: Synthesized result from adversarial reasoning

        Returns:
            Compressed final result
        """
        # Extract conclusion
        if "conclusion" in synthesis_result:
            conclusion = synthesis_result["conclusion"]
        else:
            conclusion = self._build_default_conclusion(workspace)

        # Extract key numbers (top 5 most important)
        key_numbers = self._extract_key_numbers(workspace, synthesis_result)

        # Build minimal assumptions list
        assumptions = self._compress_assumptions(workspace)

        # Get final confidence
        confidence = synthesis_result.get("confidence", workspace.global_confidence)

        # Build minimal caveats
        caveats = self._compress_caveats(workspace)

        # Suggested checks (top 3)
        suggested_checks = self._get_top_checks(workspace, synthesis_result)

        # Reasoning trace (abbreviated)
        reasoning_trace = self._build_abbreviated_trace(operator_graph)

        return {
            "conclusion": conclusion,
            "key_numbers": key_numbers,
            "assumptions": assumptions,
            "confidence": confidence,
            "caveats": caveats,
            "suggested_checks": suggested_checks,
            "reasoning_trace": reasoning_trace
        }

    def _build_default_conclusion(self, workspace: GlobalWorkspace) -> str:
        """Build default conclusion from workspace"""
        confidence = workspace.global_confidence

        if confidence > 0.8:
            return "✓ Analysis completed with high confidence."
        elif confidence > 0.5:
            return "⚠ Analysis completed with moderate confidence."
        else:
            return "✗ Analysis completed with low confidence - interpret cautiously."

    def _extract_key_numbers(
        self,
        workspace: GlobalWorkspace,
        synthesis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract top 5 most important numbers"""
        key_numbers = {}

        # Priority order
        priority_keys = ["pvalue", "f_statistic", "r_squared", "test_statistic", "confidence"]

        facts = workspace.get_all_facts()
        for key in priority_keys:
            if key in facts and len(key_numbers) < 5:
                key_numbers[key] = facts[key]

        # Add confidence
        if "confidence" not in key_numbers:
            key_numbers["confidence"] = synthesis_result.get("confidence", workspace.global_confidence)

        return key_numbers

    def _compress_assumptions(self, workspace: GlobalWorkspace) -> list[str]:
        """Get top 3 most critical assumptions"""
        assumptions = workspace.assumptions

        # Sort by criticality and confidence
        sorted_assumptions = sorted(
            assumptions,
            key=lambda a: (a.critical, -a.confidence),
            reverse=True
        )

        return [a.description for a in sorted_assumptions[:3]]

    def _compress_caveats(self, workspace: GlobalWorkspace) -> list[str]:
        """Get top 3 most important caveats"""
        caveats = []

        # Violated constraints (critical and high only)
        violated = [c for c in workspace.get_violated_constraints() if c.severity in {"HIGH", "CRITICAL"}]
        caveats.extend([c.description for c in violated[:2]])

        # Low confidence warning
        if workspace.global_confidence < 0.5:
            caveats.append("Low overall confidence in results")

        return caveats[:3]

    def _get_top_checks(
        self,
        workspace: GlobalWorkspace,
        synthesis_result: Dict[str, Any]
    ) -> list[str]:
        """Get top 3 suggested checks"""
        checks = []

        # From synthesis
        if "suggested_checks" in synthesis_result:
            checks.extend(synthesis_result["suggested_checks"][:2])

        # Default checks
        if workspace.global_confidence < 0.6:
            checks.append("Validate results with domain expert")

        if len(workspace.get_violated_constraints()) > 0:
            checks.append("Review violated constraints")

        return checks[:3]

    def _build_abbreviated_trace(self, operator_graph: OperatorGraph) -> list[str]:
        """Build abbreviated reasoning trace"""
        trace = []

        completed = [sg for sg in operator_graph.subgoals if sg.status == "completed"]
        failed = [sg for sg in operator_graph.subgoals if sg.status == "failed"]

        trace.append(f"Executed {len(operator_graph.subgoals)} subgoals")
        trace.append(f"{len(completed)} completed, {len(failed)} failed")

        # List subgoals briefly
        for sg in operator_graph.subgoals[:3]:  # Top 3 only
            status_symbol = "✓" if sg.status == "completed" else "✗"
            trace.append(f"{status_symbol} {sg.id}: {sg.description}")

        return trace
