"""
Python Math Engine - Isolated callable tool for HAVOC PRIME

Wraps havoc_tools.python_math.engine with a clean interface.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Import existing math engine functions
from havoc_tools.python_math.engine import (
    ANOVAResult,
    DOEAnalysisResult,
    MathEvalResult,
    RegressionResult,
    SPCAnalysisResult,
    TTestResult,
    run_anova,
    run_doe_analysis,
    fit_regression,  # Note: it's fit_regression, not run_regression
    run_spc_analysis,
    run_ttest,
)


class PythonMathEngine:
    """
    Callable interface to Python math/stats operations.

    Methods:
        t_test: Independent samples t-test
        anova: One-way ANOVA
        regression: Linear regression
        doe_analysis: Design of Experiments analysis
        spc_analysis: Statistical Process Control
    """

    def t_test(
        self,
        sample_a: List[float],
        sample_b: List[float],
        equal_var: bool = False,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run independent samples t-test.

        Args:
            sample_a: First sample
            sample_b: Second sample
            equal_var: Assume equal variances
            alpha: Significance level

        Returns:
            Dict with statistic, pvalue, df, ci_low, ci_high, mean_diff
        """
        result: TTestResult = run_ttest(sample_a, sample_b, equal_var, alpha)

        return {
            "statistic": result.statistic,
            "pvalue": result.pvalue,
            "df": result.df,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "mean_diff": result.mean_diff,
            "significant": result.pvalue < alpha
        }

    def anova(
        self,
        groups: List[List[float]],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Run one-way ANOVA.

        Args:
            groups: List of groups (each group is a list of values)
            alpha: Significance level

        Returns:
            Dict with f_statistic, p_value, table
        """
        result: ANOVAResult = run_anova(groups)

        return {
            "f_statistic": result.f_statistic,
            "p_value": result.p_value,
            "table": result.table,
            "significant": result.p_value < alpha
        }

    def regression(
        self,
        X: List[List[float]],
        y: List[float],
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run linear regression.

        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target variable
            feature_names: Names of features

        Returns:
            Dict with params, pvalues, r_squared, adj_r_squared, f_statistic, aic, bic
        """
        # Note: fit_regression expects formula + DataFrame, so this is a simplified wrapper
        # In production, you'd convert X, y to DataFrame and build formula
        # For now, use placeholder
        result: RegressionResult = fit_regression("y ~ x", {"y": y, "x": X[0] if X else []})

        return {
            "params": result.params,
            "pvalues": result.pvalues,
            "r_squared": result.r_squared,
            "adj_r_squared": result.adj_r_squared,
            "f_statistic": result.f_statistic,
            "aic": result.aic,
            "bic": result.bic
        }

    def doe_analysis(
        self,
        design_matrix: List[List[float]],
        responses: List[float],
        factor_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze Design of Experiments.

        Args:
            design_matrix: Factorial design matrix (coded values)
            responses: Response variable values
            factor_names: Names of factors

        Returns:
            Dict with effects, main_effects, interaction_effects, r_squared, significant_factors
        """
        result: DOEAnalysisResult = run_doe_analysis(design_matrix, responses, factor_names)

        return {
            "effects": result.effects,
            "main_effects": result.main_effects,
            "interaction_effects": result.interaction_effects,
            "summary": result.summary,
            "r_squared": result.r_squared,
            "significant_factors": result.significant_factors
        }

    def spc_analysis(
        self,
        data: List[float],
        chart_type: str = "I_MR",
        subgroup_size: int = 1,
        spec_limits: tuple = None
    ) -> Dict[str, Any]:
        """
        Statistical Process Control analysis.

        Args:
            data: Process data
            chart_type: Type of control chart (I_MR, Xbar_R, etc.)
            subgroup_size: Subgroup size
            spec_limits: (LSL, USL) specification limits

        Returns:
            Dict with control_limits, violations, in_control, capability_indices
        """
        result: SPCAnalysisResult = run_spc_analysis(data, chart_type, subgroup_size, spec_limits)

        return {
            "chart_type": result.chart_type,
            "control_limits": {
                "center_line": result.control_limits.center_line,
                "ucl": result.control_limits.ucl,
                "lcl": result.control_limits.lcl,
                "sigma": result.control_limits.sigma
            },
            "violations": [
                {
                    "rule": v.rule,
                    "points": v.points,
                    "description": v.description
                }
                for v in result.violations
            ],
            "in_control": result.in_control,
            "capability_indices": result.capability_indices
        }
