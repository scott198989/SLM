"""
Answer Formatter - Clean answer generation for HAVOC PRIME

Formats PRIME reasoning results into structured, human-readable answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FormattedAnswer:
    """
    Structured answer output.

    Attributes:
        conclusion: Main conclusion
        key_numbers: Important metrics
        assumptions: Assumptions made
        confidence: Overall confidence (0-1)
        caveats: Warnings and limitations
        suggested_checks: Recommended validations
        reasoning_trace: Step-by-step reasoning
        raw_data: Structured data for programmatic access
    """
    conclusion: str
    key_numbers: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    caveats: List[str] = field(default_factory=list)
    suggested_checks: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def format_human_readable(self) -> str:
        """Format answer as human-readable text"""
        sections = []

        # Conclusion
        sections.append("## CONCLUSION")
        sections.append(self.conclusion)
        sections.append("")

        # Confidence
        sections.append("## CONFIDENCE")
        conf_pct = self.confidence * 100
        sections.append(f"Overall: {conf_pct:.1f}%")
        sections.append("")

        # Key Numbers
        if self.key_numbers:
            sections.append("## KEY NUMBERS")
            for key, value in self.key_numbers.items():
                if isinstance(value, float):
                    sections.append(f"- {key}: {value:.4f}")
                else:
                    sections.append(f"- {key}: {value}")
            sections.append("")

        # Assumptions
        if self.assumptions:
            sections.append("## ASSUMPTIONS")
            for assumption in self.assumptions:
                sections.append(f"- {assumption}")
            sections.append("")

        # Caveats
        if self.caveats:
            sections.append("## CAVEATS")
            for caveat in self.caveats:
                sections.append(f"- {caveat}")
            sections.append("")

        # Suggested Checks
        if self.suggested_checks:
            sections.append("## SUGGESTED CHECKS")
            for check in self.suggested_checks:
                sections.append(f"- {check}")
            sections.append("")

        # Reasoning Trace
        if self.reasoning_trace:
            sections.append("## REASONING TRACE")
            for i, step in enumerate(self.reasoning_trace, 1):
                sections.append(f"{i}. {step}")
            sections.append("")

        return "\n".join(sections)


class AnswerFormatter:
    """
    Formats PRIME results into structured answers.

    Converts workspace state, subgoal results, and confidence scores
    into a comprehensive, human-readable answer.
    """

    def format_answer(
        self,
        prompt: str,
        workspace: Any,  # GlobalWorkspace
        operator_graph: Any,  # OperatorGraph
        final_result: Dict[str, Any],
        task_type: str = "GENERAL"
    ) -> FormattedAnswer:
        """
        Build formatted answer from PRIME results.

        Args:
            prompt: Original user prompt
            workspace: GlobalWorkspace instance
            operator_graph: OperatorGraph instance
            final_result: Final synthesis result
            task_type: Task type (DOE, SPC, STATS, etc.)

        Returns:
            FormattedAnswer with all sections populated
        """
        # Build conclusion
        conclusion = self._build_conclusion(final_result, workspace)

        # Extract key numbers
        key_numbers = self._extract_key_numbers(workspace, final_result)

        # Build assumptions
        assumptions = self._build_assumptions(workspace, task_type)

        # Get confidence
        confidence = workspace.global_confidence if hasattr(workspace, 'global_confidence') else 0.5

        # Build caveats
        caveats = self._build_caveats(workspace)

        # Build suggested checks
        suggested_checks = self._build_suggested_checks(task_type, confidence, workspace)

        # Build reasoning trace
        reasoning_trace = self._build_reasoning_trace(operator_graph, workspace)

        # Store raw data
        raw_data = {
            "confidence": confidence,
            "facts_count": len(workspace.facts) if hasattr(workspace, 'facts') else 0,
            "subgoals_count": len(operator_graph.subgoals) if hasattr(operator_graph, 'subgoals') else 0,
            "violated_constraints": len(workspace.get_violated_constraints()) if hasattr(workspace, 'get_violated_constraints') else 0
        }

        return FormattedAnswer(
            conclusion=conclusion,
            key_numbers=key_numbers,
            assumptions=assumptions,
            confidence=confidence,
            caveats=caveats,
            suggested_checks=suggested_checks,
            reasoning_trace=reasoning_trace,
            raw_data=raw_data
        )

    def _build_conclusion(self, final_result: Dict[str, Any], workspace: Any) -> str:
        """Build conclusion from final result"""
        if "conclusion" in final_result:
            return final_result["conclusion"]

        # Default conclusion
        confidence = workspace.global_confidence if hasattr(workspace, 'global_confidence') else 0.5

        if confidence > 0.8:
            return "✓ Analysis completed with high confidence."
        elif confidence > 0.5:
            return "⚠ Analysis completed with moderate confidence."
        else:
            return "✗ Analysis completed with low confidence - results should be interpreted cautiously."

    def _extract_key_numbers(self, workspace: Any, final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key numerical results"""
        key_numbers = {}

        # From workspace facts
        if hasattr(workspace, 'facts'):
            for fact_key, fact in workspace.facts.items():
                if isinstance(fact.value, (int, float)):
                    key_numbers[fact_key] = fact.value

        # From final result
        if "key_numbers" in final_result:
            key_numbers.update(final_result["key_numbers"])

        # Add confidence
        if hasattr(workspace, 'global_confidence'):
            key_numbers["confidence"] = workspace.global_confidence

        return key_numbers

    def _build_assumptions(self, workspace: Any, task_type: str) -> List[str]:
        """Build list of assumptions"""
        assumptions = []

        # From workspace
        if hasattr(workspace, 'assumptions'):
            assumptions.extend([a.description for a in workspace.assumptions])

        # Task-specific assumptions
        if task_type == "STATS":
            assumptions.extend([
                "Independent samples",
                "Normal distribution (or large sample size)",
                "Random sampling"
            ])
        elif task_type == "DOE":
            assumptions.extend([
                "Factors are independent",
                "Response variable is continuous",
                "No hidden factors"
            ])
        elif task_type == "SPC":
            assumptions.extend([
                "Process is stable during baseline",
                "Measurements are accurate",
                "Subgroups are rational"
            ])

        return assumptions if assumptions else ["Standard assumptions apply"]

    def _build_caveats(self, workspace: Any) -> List[str]:
        """Build list of caveats"""
        caveats = []

        # Violated constraints
        if hasattr(workspace, 'get_violated_constraints'):
            violated = workspace.get_violated_constraints()
            if violated:
                caveats.append(f"{len(violated)} constraint(s) violated")
                for constraint in violated:
                    if constraint.severity in {"HIGH", "CRITICAL"}:
                        caveats.append(f"⚠ {constraint.description}")

        # Low confidence facts
        if hasattr(workspace, 'get_low_confidence_facts'):
            low_conf_facts = workspace.get_low_confidence_facts(threshold=0.5)
            if low_conf_facts:
                caveats.append(f"{len(low_conf_facts)} fact(s) with low confidence")

        # Critical assumptions
        if hasattr(workspace, 'get_critical_assumptions'):
            critical_assumptions = workspace.get_critical_assumptions()
            if critical_assumptions:
                caveats.append(f"{len(critical_assumptions)} critical assumption(s) made")

        return caveats

    def _build_suggested_checks(self, task_type: str, confidence: float, workspace: Any) -> List[str]:
        """Build suggested validation checks"""
        checks = []

        # Low confidence warning
        if confidence < 0.5:
            checks.append("Re-examine data quality and analysis methods")
            checks.append("Consider collecting additional data")

        # Task-specific checks
        if task_type == "DOE":
            checks.extend([
                "Verify experimental design balance",
                "Check for confounding factors",
                "Validate measurement system"
            ])
        elif task_type == "SPC":
            checks.extend([
                "Review control chart rules applied",
                "Investigate out-of-control signals",
                "Verify subgroup rationale"
            ])
        elif task_type == "STATS":
            checks.extend([
                "Validate statistical assumptions",
                "Check for outliers",
                "Consider effect size, not just p-value"
            ])

        # Default checks
        if not checks:
            checks = ["Validate results with domain expert", "Replicate analysis if possible"]

        return checks

    def _build_reasoning_trace(self, operator_graph: Any, workspace: Any) -> List[str]:
        """Build step-by-step reasoning trace"""
        trace = []

        # Subgoal execution trace
        if hasattr(operator_graph, 'subgoals'):
            for sg in operator_graph.subgoals:
                status_symbol = "✓" if sg.status == "completed" else "✗" if sg.status == "failed" else "○"
                trace.append(f"{status_symbol} {sg.id}: {sg.description}")

        # Workspace history
        if hasattr(workspace, 'subgoal_history'):
            trace.append(f"Total subgoal executions: {len(workspace.subgoal_history)}")

        # Confidence evolution
        if hasattr(workspace, 'global_confidence'):
            trace.append(f"Final confidence: {workspace.global_confidence:.2f}")

        return trace if trace else ["Reasoning trace unavailable"]
