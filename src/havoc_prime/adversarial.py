"""
Adversarial Reasoning Components for HAVOC PRIME

Triple-Fork Reasoning: Advocate, HAVOC-ATTACK, Pragmatist
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from havoc_prime.workspace import GlobalWorkspace


@dataclass
class Argument:
    """An argument from one adversarial perspective"""
    perspective: str  # "ADVOCATE", "HAVOC-ATTACK", "PRAGMATIST"
    claims: List[str]
    evidence: List[Dict[str, Any]]
    confidence: float
    reasoning: str


@dataclass
class Synthesis:
    """Synthesized result from adversarial collision"""
    conclusion: str
    confidence: float
    incorporated_claims: List[str]
    rejected_claims: List[str]
    rationale: str


class Advocate:
    """
    Best-case argument builder.

    The Advocate presents the strongest positive case:
    - Highlights successful results
    - Emphasizes evidence strength
    - Downplays weaknesses
    - Optimistic interpretation
    """

    def build_argument(
        self,
        subgoal_result: Dict[str, Any],
        workspace: GlobalWorkspace
    ) -> Argument:
        """
        Build best-case argument.

        Args:
            subgoal_result: Result from subgoal execution
            workspace: Global workspace

        Returns:
            Argument with optimistic perspective
        """
        claims = []
        evidence = []
        confidence = 0.5

        # Extract positive indicators
        if "success" in subgoal_result and subgoal_result["success"]:
            claims.append("Subgoal executed successfully")
            confidence += 0.2

        # Statistical significance
        if "pvalue" in subgoal_result:
            pvalue = subgoal_result["pvalue"]
            if pvalue < 0.05:
                claims.append(f"Statistically significant result (p={pvalue:.4f})")
                evidence.append({"type": "pvalue", "value": pvalue})
                confidence += 0.3
            elif pvalue < 0.10:
                claims.append(f"Marginally significant result (p={pvalue:.4f})")
                evidence.append({"type": "pvalue", "value": pvalue})
                confidence += 0.1

        # High R-squared
        if "r_squared" in subgoal_result:
            r_sq = subgoal_result["r_squared"]
            if r_sq > 0.7:
                claims.append(f"Strong model fit (R²={r_sq:.3f})")
                evidence.append({"type": "r_squared", "value": r_sq})
                confidence += 0.2
            elif r_sq > 0.5:
                claims.append(f"Moderate model fit (R²={r_sq:.3f})")
                evidence.append({"type": "r_squared", "value": r_sq})
                confidence += 0.1

        # Control status
        if "in_control" in subgoal_result:
            if subgoal_result["in_control"]:
                claims.append("Process is in statistical control")
                confidence += 0.2

        # Workspace facts
        if workspace:
            facts = workspace.get_all_facts()
            if facts:
                claims.append(f"Analysis supported by {len(facts)} established facts")
                confidence += 0.1

        # Default claim if empty
        if not claims:
            claims = ["Analysis completed without critical errors"]
            confidence = 0.5

        # Cap confidence
        confidence = min(1.0, confidence)

        reasoning = "Best-case interpretation: " + "; ".join(claims)

        return Argument(
            perspective="ADVOCATE",
            claims=claims,
            evidence=evidence,
            confidence=confidence,
            reasoning=reasoning
        )


class HavocAttack:
    """
    Aggressive critic (Devil's Advocate on steroids).

    HAVOC-ATTACK tries to destroy the conclusion:
    - Identifies weaknesses
    - Questions assumptions
    - Highlights violations
    - Pessimistic interpretation
    - Searches for brittleness
    """

    def build_argument(
        self,
        subgoal_result: Dict[str, Any],
        workspace: GlobalWorkspace
    ) -> Argument:
        """
        Build critical argument.

        Args:
            subgoal_result: Result from subgoal execution
            workspace: Global workspace

        Returns:
            Argument with critical perspective
        """
        claims = []
        evidence = []
        confidence = 0.5

        # Extract weaknesses
        if "success" in subgoal_result and not subgoal_result["success"]:
            claims.append("Subgoal execution failed")
            confidence += 0.3

        if "error" in subgoal_result:
            claims.append(f"Execution error: {subgoal_result['error']}")
            evidence.append({"type": "error", "value": subgoal_result["error"]})
            confidence += 0.4

        # Statistical insignificance
        if "pvalue" in subgoal_result:
            pvalue = subgoal_result["pvalue"]
            if pvalue >= 0.05:
                claims.append(f"NOT statistically significant (p={pvalue:.4f})")
                evidence.append({"type": "pvalue", "value": pvalue})
                confidence += 0.3

        # Low R-squared
        if "r_squared" in subgoal_result:
            r_sq = subgoal_result["r_squared"]
            if r_sq < 0.5:
                claims.append(f"Weak model fit (R²={r_sq:.3f})")
                evidence.append({"type": "r_squared", "value": r_sq})
                confidence += 0.2

        # Out of control
        if "in_control" in subgoal_result:
            if not subgoal_result["in_control"]:
                claims.append("Process is OUT OF CONTROL")
                confidence += 0.3

        if "violations" in subgoal_result:
            violations = subgoal_result["violations"]
            if violations:
                claims.append(f"Found {len(violations)} control violations")
                evidence.append({"type": "violations", "count": len(violations)})
                confidence += 0.2

        # Workspace issues
        if workspace:
            violated_constraints = workspace.get_violated_constraints()
            if violated_constraints:
                claims.append(f"{len(violated_constraints)} constraint(s) violated")
                confidence += 0.2

            low_conf_facts = workspace.get_low_confidence_facts(threshold=0.5)
            if low_conf_facts:
                claims.append(f"{len(low_conf_facts)} fact(s) with low confidence")
                confidence += 0.1

            critical_assumptions = workspace.get_critical_assumptions()
            if critical_assumptions:
                claims.append(f"Relies on {len(critical_assumptions)} critical assumption(s)")
                confidence += 0.1

        # Default claim if empty
        if not claims:
            claims = ["Insufficient evidence for strong conclusion"]
            confidence = 0.4

        # Cap confidence
        confidence = min(1.0, confidence)

        reasoning = "Critical analysis: " + "; ".join(claims)

        return Argument(
            perspective="HAVOC-ATTACK",
            claims=claims,
            evidence=evidence,
            confidence=confidence,
            reasoning=reasoning
        )


class Pragmatist:
    """
    Practical feasibility checker.

    The Pragmatist asks "Can this actually work?":
    - Assesses practical constraints
    - Considers resource limitations
    - Evaluates real-world applicability
    - Balanced interpretation
    """

    def build_argument(
        self,
        subgoal_result: Dict[str, Any],
        workspace: GlobalWorkspace
    ) -> Argument:
        """
        Build pragmatic argument.

        Args:
            subgoal_result: Result from subgoal execution
            workspace: Global workspace

        Returns:
            Argument with pragmatic perspective
        """
        claims = []
        evidence = []
        confidence = 0.5

        # Practical considerations
        if "success" in subgoal_result:
            if subgoal_result["success"]:
                claims.append("Result is technically sound")
                confidence += 0.15
            else:
                claims.append("Result has technical issues")
                confidence -= 0.1

        # Balance statistical and practical significance
        if "pvalue" in subgoal_result and "effect_size" in subgoal_result:
            pvalue = subgoal_result["pvalue"]
            effect_size = subgoal_result["effect_size"]

            if pvalue < 0.05 and effect_size > 0.5:
                claims.append("Both statistically and practically significant")
                confidence += 0.3
            elif pvalue < 0.05 and effect_size < 0.2:
                claims.append("Statistically significant but effect is small")
                evidence.append({"type": "small_effect", "pvalue": pvalue, "effect_size": effect_size})
                confidence += 0.1

        elif "pvalue" in subgoal_result:
            # No effect size - pragmatic concern
            claims.append("Statistical significance without effect size context")
            confidence += 0.05

        # Resource constraints
        if "sample_size" in subgoal_result:
            n = subgoal_result["sample_size"]
            if n < 30:
                claims.append(f"Small sample size (n={n}) - results may not generalize")
                evidence.append({"type": "sample_size", "value": n})
                confidence -= 0.1

        # Practical applicability
        if workspace:
            facts = workspace.get_all_facts()
            if facts:
                claims.append(f"Grounded in {len(facts)} established facts")
                confidence += 0.1

            # Check if assumptions are reasonable
            assumptions = workspace.assumptions
            critical_assumptions = [a for a in assumptions if a.critical]
            if critical_assumptions:
                claims.append(f"Depends on {len(critical_assumptions)} critical assumptions - verify before deploying")
                confidence -= 0.05

        # Default claim
        if not claims:
            claims = ["Results appear reasonable but require validation"]
            confidence = 0.5

        # Cap confidence
        confidence = max(0.0, min(1.0, confidence))

        reasoning = "Pragmatic assessment: " + "; ".join(claims)

        return Argument(
            perspective="PRAGMATIST",
            claims=claims,
            evidence=evidence,
            confidence=confidence,
            reasoning=reasoning
        )


class AdversarialSynthesizer:
    """
    Synthesizes results from triple-fork reasoning.

    Takes Advocate, HAVOC-ATTACK, and Pragmatist arguments and
    produces a unified, battle-tested conclusion.
    """

    def synthesize(
        self,
        advocate_arg: Argument,
        attack_arg: Argument,
        pragmatist_arg: Argument
    ) -> Synthesis:
        """
        Synthesize three perspectives into unified conclusion.

        Args:
            advocate_arg: Optimistic argument
            attack_arg: Critical argument
            pragmatist_arg: Pragmatic argument

        Returns:
            Synthesis with final conclusion
        """
        incorporated_claims = []
        rejected_claims = []

        # Weight perspectives
        weights = {
            "ADVOCATE": 0.3,
            "HAVOC-ATTACK": 0.4,  # Give critic more weight (pessimistic bias)
            "PRAGMATIST": 0.3
        }

        # Weighted confidence
        final_confidence = (
            weights["ADVOCATE"] * advocate_arg.confidence +
            weights["HAVOC-ATTACK"] * (1.0 - attack_arg.confidence) +  # Invert attack confidence
            weights["PRAGMATIST"] * pragmatist_arg.confidence
        )

        # Build conclusion
        if final_confidence > 0.7:
            conclusion_prefix = "✓ Strong evidence supports the conclusion."
        elif final_confidence > 0.5:
            conclusion_prefix = "⚠ Moderate evidence supports the conclusion with caveats."
        else:
            conclusion_prefix = "✗ Insufficient evidence for strong conclusion."

        # Incorporate claims from all perspectives
        # Accept strong claims from Advocate
        for claim in advocate_arg.claims:
            if advocate_arg.confidence > 0.6:
                incorporated_claims.append(f"[ADVOCATE] {claim}")

        # Accept critical claims from HAVOC-ATTACK (these are important warnings)
        for claim in attack_arg.claims:
            if attack_arg.confidence > 0.5:
                incorporated_claims.append(f"[CRITIC] {claim}")

        # Accept balanced claims from Pragmatist
        for claim in pragmatist_arg.claims:
            if pragmatist_arg.confidence > 0.4:
                incorporated_claims.append(f"[PRAGMATIC] {claim}")

        # Build rationale
        rationale_parts = [
            f"Advocate confidence: {advocate_arg.confidence:.2f}",
            f"Critic confidence: {attack_arg.confidence:.2f}",
            f"Pragmatist confidence: {pragmatist_arg.confidence:.2f}",
            f"Final weighted confidence: {final_confidence:.2f}"
        ]

        rationale = " | ".join(rationale_parts)

        conclusion = f"{conclusion_prefix}\n\nKey considerations:\n" + "\n".join(f"- {c}" for c in incorporated_claims[:5])

        return Synthesis(
            conclusion=conclusion,
            confidence=final_confidence,
            incorporated_claims=incorporated_claims,
            rejected_claims=rejected_claims,
            rationale=rationale
        )
