from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from havoc_srs.execute import ExecutionResult


@dataclass
class Argument:
    direction: str
    evidence: List[str]
    confidence: float
    text: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)


def build_arguments(prompt: str, exec_result: ExecutionResult = None) -> List[Argument]:
    """Build PRO and CON arguments based on execution results."""
    pro_evidence = []
    con_evidence = []
    pro_confidence = 0.5
    con_confidence = 0.5
    pro_data = {}
    con_data = {}

    if exec_result:
        # Analyze execution results to build arguments
        for step in exec_result.steps:
            if step.success:
                pro_evidence.append(f"{step.step_description}: Success")

                # Extract specific evidence from outputs
                if "payload" in step.output:
                    payload = step.output["payload"]

                    # Statistical significance
                    if "pvalue" in payload:
                        pvalue = payload["pvalue"]
                        if pvalue < 0.05:
                            pro_evidence.append(f"Statistically significant (p={pvalue:.4f})")
                            pro_confidence += 0.2
                            pro_data["pvalue"] = pvalue
                        else:
                            con_evidence.append(f"Not statistically significant (p={pvalue:.4f})")
                            con_confidence += 0.15
                            con_data["pvalue"] = pvalue

                    # SPC control
                    if "in_control" in payload:
                        in_control = payload["in_control"]
                        if in_control:
                            pro_evidence.append("Process is in statistical control")
                            pro_confidence += 0.15
                            pro_data["in_control"] = True
                        else:
                            con_evidence.append("Process shows out-of-control signals")
                            con_confidence += 0.2
                            violations = payload.get("violations", [])
                            if violations:
                                con_evidence.append(f"Found {len(violations)} control rule violations")
                            con_data["in_control"] = False
                            con_data["violations"] = len(violations)

                    # DOE significant factors
                    if "significant_factors" in payload:
                        sig_factors = payload["significant_factors"]
                        if sig_factors:
                            pro_evidence.append(f"Identified {len(sig_factors)} significant factors: {', '.join(sig_factors)}")
                            pro_confidence += 0.15
                            pro_data["significant_factors"] = sig_factors
                        else:
                            con_evidence.append("No significant factors identified")
                            con_confidence += 0.1

                    # R-squared for model fit
                    if "r_squared" in payload:
                        r_sq = payload["r_squared"]
                        if r_sq > 0.7:
                            pro_evidence.append(f"Good model fit (R²={r_sq:.3f})")
                            pro_confidence += 0.1
                            pro_data["r_squared"] = r_sq
                        elif r_sq < 0.3:
                            con_evidence.append(f"Poor model fit (R²={r_sq:.3f})")
                            con_confidence += 0.15
                            con_data["r_squared"] = r_sq

            else:
                # Execution failed
                con_evidence.append(f"{step.step_description}: Failed")
                con_confidence += 0.2
                if step.error:
                    con_evidence.append(f"Error: {step.error}")
                    con_data["error"] = step.error

        # Adjust based on overall execution success
        if exec_result.overall_success:
            pro_confidence += 0.1
            pro_evidence.append("All execution steps succeeded")
        else:
            con_confidence += 0.15
            con_evidence.append("Some execution steps failed")

    # Default arguments if no execution result
    if not pro_evidence:
        pro_evidence = ["Analysis completed without errors"]
        pro_confidence = 0.6

    if not con_evidence:
        con_evidence = ["Limited evidence against conclusion"]
        con_confidence = 0.4

    # Normalize confidences to [0, 1]
    pro_confidence = min(1.0, max(0.0, pro_confidence))
    con_confidence = min(1.0, max(0.0, con_confidence))

    # Build argument texts
    pro_text = "PRO: " + "; ".join(pro_evidence)
    con_text = "CON: " + "; ".join(con_evidence)

    pro = Argument(
        direction="PRO",
        evidence=pro_evidence,
        confidence=pro_confidence,
        text=pro_text,
        supporting_data=pro_data
    )

    con = Argument(
        direction="CON",
        evidence=con_evidence,
        confidence=con_confidence,
        text=con_text,
        supporting_data=con_data
    )

    return [pro, con]
