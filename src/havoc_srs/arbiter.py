from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from havoc_srs.argue import Argument


@dataclass
class ArbiterDecision:
    winner: str
    confidence: float
    rationale: List[str]
    consistency_score: float = 1.0
    data_quality_score: float = 1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)


def decide(arguments: List[Argument]) -> ArbiterDecision:
    """Decide between arguments using confidence, consistency, and data quality checks."""
    if not arguments:
        return ArbiterDecision(
            winner="UNDECIDED",
            confidence=0.0,
            rationale=["No arguments provided"],
            consistency_score=0.0,
            data_quality_score=0.0
        )

    # Find highest confidence argument
    winner_arg = max(arguments, key=lambda a: a.confidence)

    # Build rationale
    rationale = [winner_arg.text]

    # Perform consistency checks
    consistency_score, consistency_notes = _check_consistency(arguments)
    rationale.extend(consistency_notes)

    # Perform data quality checks
    data_quality_score, quality_notes = _check_data_quality(winner_arg)
    rationale.extend(quality_notes)

    # Adjust confidence based on consistency and data quality
    adjusted_confidence = winner_arg.confidence * consistency_score * data_quality_score

    # Extract supporting data from winner
    supporting_data = winner_arg.supporting_data.copy()
    supporting_data["original_confidence"] = winner_arg.confidence
    supporting_data["consistency_score"] = consistency_score
    supporting_data["data_quality_score"] = data_quality_score

    return ArbiterDecision(
        winner=winner_arg.direction,
        confidence=adjusted_confidence,
        rationale=rationale,
        consistency_score=consistency_score,
        data_quality_score=data_quality_score,
        supporting_data=supporting_data
    )


def _check_consistency(arguments: List[Argument]) -> tuple[float, List[str]]:
    """Check consistency between arguments."""
    notes = []
    score = 1.0

    if len(arguments) < 2:
        return score, notes

    # Get PRO and CON arguments
    pro_args = [a for a in arguments if a.direction == "PRO"]
    con_args = [a for a in arguments if a.direction == "CON"]

    if not pro_args or not con_args:
        return score, notes

    pro_conf = pro_args[0].confidence
    con_conf = con_args[0].confidence

    # Check for conflicting evidence
    pro_data = pro_args[0].supporting_data
    con_data = con_args[0].supporting_data

    # If both have similar confidence, it's inconsistent
    conf_diff = abs(pro_conf - con_conf)
    if conf_diff < 0.2:
        score *= 0.7
        notes.append(f"Conflicting evidence: PRO conf={pro_conf:.2f}, CON conf={con_conf:.2f}")

    # Check for contradictory statistical results
    if "pvalue" in pro_data and "pvalue" in con_data:
        # This would indicate internal inconsistency
        score *= 0.5
        notes.append("Inconsistent: Both PRO and CON claim statistical significance")

    if "in_control" in pro_data and "in_control" in con_data:
        if pro_data["in_control"] != con_data["in_control"]:
            # This shouldn't happen - indicates error
            score *= 0.3
            notes.append("Critical inconsistency: Contradictory control status")

    if score < 1.0:
        notes.append(f"Consistency score: {score:.2f}")
    else:
        notes.append("Arguments are consistent")

    return score, notes


def _check_data_quality(argument: Argument) -> tuple[float, List[str]]:
    """Check data quality of the winning argument."""
    notes = []
    score = 1.0

    data = argument.supporting_data

    # Check for errors
    if "error" in data:
        score *= 0.3
        notes.append(f"Data quality issue: {data['error']}")

    # Check for weak statistical evidence
    if "pvalue" in data:
        pvalue = data["pvalue"]
        if pvalue > 0.1:
            score *= 0.8
            notes.append(f"Weak statistical evidence (p={pvalue:.4f})")
        elif pvalue < 0.01:
            notes.append(f"Strong statistical evidence (p={pvalue:.4f})")

    # Check for poor model fit
    if "r_squared" in data:
        r_sq = data["r_squared"]
        if r_sq < 0.5:
            score *= 0.7
            notes.append(f"Low model fit (R²={r_sq:.3f})")
        elif r_sq > 0.8:
            notes.append(f"Excellent model fit (R²={r_sq:.3f})")

    # Check for control violations (if SPC)
    if "violations" in data:
        violations = data["violations"]
        if violations > 5:
            score *= 0.6
            notes.append(f"Multiple control violations ({violations})")

    # If no evidence was found
    if not data or len(data) == 0:
        score *= 0.7
        notes.append("Limited supporting data")

    if score < 1.0:
        notes.append(f"Data quality score: {score:.2f}")
    else:
        notes.append("Data quality is good")

    return score, notes
