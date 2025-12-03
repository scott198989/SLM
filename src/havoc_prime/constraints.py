"""
Constraint Backbone for HAVOC PRIME

Domain-specific constraints enforced throughout reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List

from havoc_prime.router import TaskType
from havoc_prime.workspace import GlobalWorkspace


class ConstraintBackbone:
    """
    Enforces domain-specific constraints across all subgoals.

    Constraints are rules that must be satisfied for valid reasoning.
    Violations downgrade confidence and are reported as caveats.
    """

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.constraints = self._initialize_constraints()

    def _initialize_constraints(self) -> List[Dict[str, Any]]:
        """Initialize task-specific constraints"""
        if self.task_type == TaskType.STATS:
            return [
                {
                    "name": "sample_size_check",
                    "description": "Sample size must be adequate (n >= 30 for normality)",
                    "checker": self._check_sample_size,
                    "severity": "MEDIUM"
                },
                {
                    "name": "assumption_validation",
                    "description": "Statistical assumptions must be validated",
                    "checker": self._check_assumptions_validated,
                    "severity": "HIGH"
                },
                {
                    "name": "effect_size_reporting",
                    "description": "Effect size must be reported alongside p-value",
                    "checker": self._check_effect_size,
                    "severity": "MEDIUM"
                }
            ]

        elif self.task_type == TaskType.DOE:
            return [
                {
                    "name": "design_balance",
                    "description": "Design must be balanced (equal replicates)",
                    "checker": self._check_design_balance,
                    "severity": "HIGH"
                },
                {
                    "name": "confounding_check",
                    "description": "Factors must not be confounded",
                    "checker": self._check_confounding,
                    "severity": "CRITICAL"
                },
                {
                    "name": "factor_independence",
                    "description": "Factors must be independent",
                    "checker": self._check_factor_independence,
                    "severity": "HIGH"
                }
            ]

        elif self.task_type == TaskType.SPC:
            return [
                {
                    "name": "baseline_stability",
                    "description": "Baseline period must be in control",
                    "checker": self._check_baseline_stability,
                    "severity": "CRITICAL"
                },
                {
                    "name": "subgroup_rationality",
                    "description": "Subgroups must be rational",
                    "checker": self._check_subgroup_rationality,
                    "severity": "HIGH"
                },
                {
                    "name": "measurement_accuracy",
                    "description": "Measurement system must be validated",
                    "checker": self._check_measurement_system,
                    "severity": "HIGH"
                }
            ]

        else:
            return []

    def enforce(self, workspace: GlobalWorkspace) -> List[str]:
        """
        Enforce all constraints and return violations.

        Args:
            workspace: Global workspace with facts

        Returns:
            List of violation descriptions
        """
        violations = []

        for constraint in self.constraints:
            checker = constraint["checker"]
            is_valid = checker(workspace)

            if not is_valid:
                violation_msg = f"{constraint['severity']}: {constraint['description']}"
                violations.append(violation_msg)

                # Mark constraint as violated in workspace
                workspace.add_constraint(constraint["description"], severity=constraint["severity"])
                workspace.mark_constraint_violated(constraint["description"])

        return violations

    # Constraint checkers (placeholders - would be implemented with actual logic)

    def _check_sample_size(self, workspace: GlobalWorkspace) -> bool:
        """Check if sample size is adequate"""
        sample_size_fact = workspace.get_fact("sample_size")
        if sample_size_fact:
            return sample_size_fact.value >= 30
        return True  # Pass if no sample size specified

    def _check_assumptions_validated(self, workspace: GlobalWorkspace) -> bool:
        """Check if statistical assumptions were validated"""
        assumptions_checked = workspace.get_fact("assumptions_validated")
        return assumptions_checked.value if assumptions_checked else False

    def _check_effect_size(self, workspace: GlobalWorkspace) -> bool:
        """Check if effect size is reported"""
        effect_size = workspace.get_fact("effect_size")
        return effect_size is not None

    def _check_design_balance(self, workspace: GlobalWorkspace) -> bool:
        """Check if DOE design is balanced"""
        design_balanced = workspace.get_fact("design_balanced")
        return design_balanced.value if design_balanced else True

    def _check_confounding(self, workspace: GlobalWorkspace) -> bool:
        """Check for confounding"""
        confounding = workspace.get_fact("confounding_detected")
        return not confounding.value if confounding else True

    def _check_factor_independence(self, workspace: GlobalWorkspace) -> bool:
        """Check factor independence"""
        factors_independent = workspace.get_fact("factors_independent")
        return factors_independent.value if factors_independent else True

    def _check_baseline_stability(self, workspace: GlobalWorkspace) -> bool:
        """Check SPC baseline stability"""
        baseline_stable = workspace.get_fact("baseline_stable")
        return baseline_stable.value if baseline_stable else True

    def _check_subgroup_rationality(self, workspace: GlobalWorkspace) -> bool:
        """Check subgroup rationality"""
        subgroups_rational = workspace.get_fact("subgroups_rational")
        return subgroups_rational.value if subgroups_rational else True

    def _check_measurement_system(self, workspace: GlobalWorkspace) -> bool:
        """Check measurement system validation"""
        measurement_validated = workspace.get_fact("measurement_validated")
        return measurement_validated.value if measurement_validated else True
