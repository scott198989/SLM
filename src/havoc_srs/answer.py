from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from havoc_srs.audit import AuditReport
from havoc_srs.arbiter import ArbiterDecision
from havoc_srs.execute import ExecutionResult
from havoc_srs.ground import GroundedContext
from havoc_srs.mode import ModePrediction


@dataclass
class Answer:
    conclusion: str
    key_numbers: Dict[str, Any]
    assumptions: List[str]
    confidence: float
    caveats: List[str]
    suggested_checks: List[str]
    reasoning_trace: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def format_human_readable(self) -> str:
        """Format answer as human-readable text."""
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


def build_answer(
    prompt: str,
    decision: ArbiterDecision,
    audit: AuditReport,
    exec_result: ExecutionResult,
    grounded: GroundedContext,
    mode: ModePrediction = None,
) -> Answer:
    """Build comprehensive human-readable answer from SRS pipeline results."""

    # Build conclusion based on decision and execution results
    conclusion_parts = []

    if decision.winner == "PRO":
        conclusion_parts.append("✓ Analysis supports the hypothesis.")
    elif decision.winner == "CON":
        conclusion_parts.append("✗ Analysis does not support the hypothesis.")
    else:
        conclusion_parts.append("⚠ Unable to reach a definitive conclusion.")

    # Add execution summary
    if exec_result:
        conclusion_parts.append(f"\n{exec_result.summary}")

    # Add decision rationale (first line only for brevity)
    if decision.rationale:
        conclusion_parts.append(f"\nRationale: {decision.rationale[0]}")

    conclusion = "\n".join(conclusion_parts)

    # Extract key numbers from execution results and decision
    key_numbers = {}

    if exec_result:
        for step in exec_result.steps:
            if step.success and "payload" in step.output:
                payload = step.output["payload"]

                # Extract statistical measures
                if "pvalue" in payload:
                    key_numbers["p_value"] = payload["pvalue"]
                if "statistic" in payload:
                    key_numbers["test_statistic"] = payload["statistic"]
                if "r_squared" in payload:
                    key_numbers["r_squared"] = payload["r_squared"]
                if "f_statistic" in payload:
                    key_numbers["f_statistic"] = payload["f_statistic"]

                # Extract control limits
                if "control_limits" in payload:
                    cl = payload["control_limits"]
                    key_numbers["center_line"] = cl.get("center_line", cl.get("cl"))
                    key_numbers["ucl"] = cl.get("ucl")
                    key_numbers["lcl"] = cl.get("lcl")

                # Extract significant factors
                if "significant_factors" in payload:
                    key_numbers["significant_factors_count"] = len(payload["significant_factors"])

    # Add confidence metrics
    key_numbers["final_confidence"] = audit.downgraded_confidence
    if decision.supporting_data:
        if "original_confidence" in decision.supporting_data:
            key_numbers["initial_confidence"] = decision.supporting_data["original_confidence"]
        if "consistency_score" in decision.supporting_data:
            key_numbers["consistency_score"] = decision.supporting_data["consistency_score"]
        if "data_quality_score" in decision.supporting_data:
            key_numbers["data_quality_score"] = decision.supporting_data["data_quality_score"]

    # Build assumptions list
    assumptions = []

    if mode:
        assumptions.append(f"Task classified as: {mode.task.name}")
        assumptions.append(f"Difficulty: {mode.difficulty.name}")
        assumptions.append(f"Risk: {mode.risk.name}")

    # Add data-specific assumptions
    if exec_result:
        for step in exec_result.steps:
            if "ttest" in step.step_description.lower():
                assumptions.append("Independent samples with normal distributions")
            elif "anova" in step.step_description.lower():
                assumptions.append("Equal variances across groups (homoscedasticity)")
                assumptions.append("Independent observations")
            elif "regression" in step.step_description.lower():
                assumptions.append("Linear relationship between variables")
                assumptions.append("Residuals are normally distributed")
            elif "doe" in step.step_description.lower():
                assumptions.append("Factors are independent")
                assumptions.append("Response variable is continuous")
            elif "spc" in step.step_description.lower():
                assumptions.append("Process is stable over time")
                assumptions.append("Measurements are accurate")

    if not assumptions:
        assumptions = ["Standard statistical assumptions apply"]

    # Caveats from audit
    caveats = audit.issues.copy()

    # Add caveats from decision
    if decision.consistency_score < 0.8:
        caveats.append("Low consistency between arguments")
    if decision.data_quality_score < 0.7:
        caveats.append("Data quality concerns identified")

    # Add caveats from execution failures
    if exec_result and not exec_result.overall_success:
        caveats.append("Some analysis steps failed - results may be incomplete")

    # Suggested checks based on task type and results
    suggested_checks = []

    if mode:
        if mode.task.name == "DOE":
            suggested_checks.append("Verify experimental design balance")
            suggested_checks.append("Check for confounding factors")
            suggested_checks.append("Validate measurement system")
        elif mode.task.name == "SPC":
            suggested_checks.append("Review control chart rules applied")
            suggested_checks.append("Investigate out-of-control signals")
            suggested_checks.append("Verify subgroup rationale")
        elif mode.task.name == "STATS":
            suggested_checks.append("Validate statistical assumptions")
            suggested_checks.append("Check for outliers")
            suggested_checks.append("Consider effect size, not just p-value")

    # Add check for low confidence
    if audit.downgraded_confidence < 0.5:
        suggested_checks.append("Re-examine data quality and analysis methods")

    # Add RAG-based checks
    if grounded and grounded.references:
        suggested_checks.append(f"Review {len(grounded.references)} retrieved references for context")

    if not suggested_checks:
        suggested_checks = ["Validate results with domain expert", "Replicate analysis if possible"]

    # Build reasoning trace
    reasoning_trace = [
        f"MODE: Classified problem as {mode.task.name if mode else 'UNKNOWN'}",
        f"GROUND: Retrieved {len(grounded.references) if grounded else 0} references",
        f"PLAN: Created {len(exec_result.steps) if exec_result else 0} execution steps",
        f"EXECUTE: {exec_result.summary if exec_result else 'No execution'}",
        f"ARGUE: Built arguments with confidence PRO vs CON",
        f"ARBITER: Selected {decision.winner} (confidence={decision.confidence:.2f})",
        f"AUDIT: {audit.severity} severity, {len(audit.issues)} issues",
        f"ANSWER: Final confidence = {audit.downgraded_confidence:.2f}"
    ]

    # Store raw data for programmatic access
    raw_data = {
        "decision": {
            "winner": decision.winner,
            "confidence": decision.confidence,
            "consistency_score": decision.consistency_score,
            "data_quality_score": decision.data_quality_score,
        },
        "audit": {
            "issues": audit.issues,
            "severity": audit.severity,
            "downgraded_confidence": audit.downgraded_confidence,
        },
        "execution": {
            "overall_success": exec_result.overall_success if exec_result else False,
            "steps": len(exec_result.steps) if exec_result else 0,
        }
    }

    return Answer(
        conclusion=conclusion,
        key_numbers=key_numbers,
        assumptions=assumptions,
        confidence=audit.downgraded_confidence,
        caveats=caveats,
        suggested_checks=suggested_checks,
        reasoning_trace=reasoning_trace,
        raw_data=raw_data,
    )
