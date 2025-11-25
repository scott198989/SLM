from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sympy import Symbol, diff, sympify


@dataclass
class TTestResult:
    statistic: float
    pvalue: float
    df: float
    ci_low: float
    ci_high: float
    mean_diff: float


@dataclass
class ANOVAResult:
    table: Dict[str, Any]
    f_statistic: float
    p_value: float


@dataclass
class RegressionResult:
    params: Dict[str, float]
    pvalues: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    aic: float
    bic: float


@dataclass
class DOEAnalysisResult:
    effects: Dict[str, float]
    main_effects: Dict[str, float]
    interaction_effects: Dict[str, float]
    summary: str
    r_squared: float
    significant_factors: List[str]


@dataclass
class SPCControlLimits:
    center_line: float
    ucl: float
    lcl: float
    sigma: float


@dataclass
class SPCViolation:
    rule: str
    points: List[int]
    description: str


@dataclass
class SPCAnalysisResult:
    chart_type: str
    control_limits: SPCControlLimits
    violations: List[SPCViolation]
    in_control: bool
    capability_indices: Optional[Dict[str, float]] = None


@dataclass
class MathEvalResult:
    result: float
    expression: str
    variables: Dict[str, float]


def run_ttest(sample_a: List[float], sample_b: List[float], equal_var: bool = False, alpha: float = 0.05) -> TTestResult:
    """Run independent samples t-test with confidence interval."""
    a_arr = np.array(sample_a)
    b_arr = np.array(sample_b)

    stat, pval = stats.ttest_ind(a_arr, b_arr, equal_var=equal_var)
    df = len(sample_a) + len(sample_b) - 2
    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))

    # Calculate confidence interval for mean difference
    if equal_var:
        pooled_std = np.sqrt(((len(a_arr) - 1) * np.var(a_arr, ddof=1) +
                              (len(b_arr) - 1) * np.var(b_arr, ddof=1)) / df)
        se = pooled_std * np.sqrt(1/len(a_arr) + 1/len(b_arr))
    else:
        se = np.sqrt(np.var(a_arr, ddof=1)/len(a_arr) + np.var(b_arr, ddof=1)/len(b_arr))

    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    return TTestResult(
        statistic=float(stat),
        pvalue=float(pval),
        df=df,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        mean_diff=mean_diff
    )


def run_anova(formula: str, data) -> ANOVAResult:
    """Run ANOVA analysis with detailed results."""
    model = smf.ols(formula=formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract F-statistic and p-value for main effect
    f_stat = float(anova_table['F'].iloc[0])
    p_val = float(anova_table['PR(>F)'].iloc[0])

    return ANOVAResult(
        table=anova_table.to_dict(),
        f_statistic=f_stat,
        p_value=p_val
    )


def fit_regression(formula: str, data) -> RegressionResult:
    """Fit regression model with comprehensive diagnostics."""
    model = smf.ols(formula=formula, data=data).fit()
    return RegressionResult(
        params=dict(model.params),
        pvalues=dict(model.pvalues),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue),
        aic=float(model.aic),
        bic=float(model.bic)
    )


def run_doe_analysis(design_matrix: np.ndarray, responses: np.ndarray, factor_names: Optional[List[str]] = None) -> DOEAnalysisResult:
    """Run DOE analysis with main effects and interactions."""
    if factor_names is None:
        factor_names = [f"X{i+1}" for i in range(design_matrix.shape[1])]

    X = sm.add_constant(design_matrix)
    model = sm.OLS(responses, X).fit()

    # Main effects (excluding intercept)
    main_effects = {factor_names[i]: float(model.params[i+1])
                   for i in range(len(factor_names))}

    # Identify significant factors (p < 0.05)
    significant_factors = [factor_names[i] for i in range(len(factor_names))
                          if model.pvalues[i+1] < 0.05]

    # All effects including intercept
    effects = {"Intercept": float(model.params[0])}
    effects.update(main_effects)

    # Interaction effects (placeholder - would need interaction terms in design)
    interaction_effects = {}

    return DOEAnalysisResult(
        effects=effects,
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        summary=model.summary().as_text(),
        r_squared=float(model.rsquared),
        significant_factors=significant_factors
    )


def calculate_control_limits_xbar_r(data: np.ndarray, subgroup_size: int) -> SPCControlLimits:
    """Calculate control limits for X-bar and R chart."""
    n_subgroups = len(data) // subgroup_size
    subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    xbar = np.mean(subgroups, axis=1)
    r = np.ptp(subgroups, axis=1)  # Range

    xbar_mean = np.mean(xbar)
    r_mean = np.mean(r)

    # Constants for X-bar chart
    A2_values = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
    A2 = A2_values.get(subgroup_size, 0.577)

    # Estimate sigma
    d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
    d2 = d2_values.get(subgroup_size, 2.326)
    sigma = r_mean / d2

    ucl = xbar_mean + A2 * r_mean
    lcl = xbar_mean - A2 * r_mean

    return SPCControlLimits(center_line=xbar_mean, ucl=ucl, lcl=lcl, sigma=sigma)


def check_weco_rules(data: np.ndarray, control_limits: SPCControlLimits, rules: List[str]) -> List[SPCViolation]:
    """Check Western Electric Company (WECO) rules for control chart violations."""
    violations = []
    cl = control_limits.center_line
    ucl = control_limits.ucl
    lcl = control_limits.lcl
    sigma = control_limits.sigma

    # WECO Rule 1: Any point beyond 3-sigma
    if "WECO_1" in rules:
        beyond_3sigma = np.where((data > ucl) | (data < lcl))[0]
        if len(beyond_3sigma) > 0:
            violations.append(SPCViolation(
                rule="WECO_1",
                points=beyond_3sigma.tolist(),
                description="Point(s) beyond 3-sigma control limits"
            ))

    # WECO Rule 2: 2 out of 3 consecutive points beyond 2-sigma
    if "WECO_2" in rules:
        ucl_2s = cl + 2 * sigma
        lcl_2s = cl - 2 * sigma
        for i in range(len(data) - 2):
            window = data[i:i+3]
            beyond_2s = np.sum((window > ucl_2s) | (window < lcl_2s))
            if beyond_2s >= 2:
                violations.append(SPCViolation(
                    rule="WECO_2",
                    points=list(range(i, i+3)),
                    description=f"2 of 3 consecutive points beyond 2-sigma (points {i}-{i+2})"
                ))

    # WECO Rule 3: 4 out of 5 consecutive points beyond 1-sigma
    if "WECO_3" in rules:
        ucl_1s = cl + sigma
        lcl_1s = cl - sigma
        for i in range(len(data) - 4):
            window = data[i:i+5]
            beyond_1s = np.sum((window > ucl_1s) | (window < lcl_1s))
            if beyond_1s >= 4:
                violations.append(SPCViolation(
                    rule="WECO_3",
                    points=list(range(i, i+5)),
                    description=f"4 of 5 consecutive points beyond 1-sigma (points {i}-{i+4})"
                ))

    # WECO Rule 4: 8 consecutive points on one side of centerline
    if "WECO_4" in rules:
        for i in range(len(data) - 7):
            window = data[i:i+8]
            all_above = np.all(window > cl)
            all_below = np.all(window < cl)
            if all_above or all_below:
                violations.append(SPCViolation(
                    rule="WECO_4",
                    points=list(range(i, i+8)),
                    description=f"8 consecutive points on one side of centerline (points {i}-{i+7})"
                ))

    return violations


def run_spc_analysis(
    chart_type: str,
    data: List[float],
    subgroup_size: Optional[int] = None,
    rules: List[str] = None,
    target: Optional[float] = None
) -> SPCAnalysisResult:
    """Run SPC control chart analysis."""
    if rules is None:
        rules = ["WECO_1"]

    data_arr = np.array(data)

    # Calculate control limits based on chart type
    if chart_type in ["XBar_R", "XBar_S"]:
        if subgroup_size is None:
            raise ValueError("subgroup_size required for XBar charts")
        control_limits = calculate_control_limits_xbar_r(data_arr, subgroup_size)

        # For analysis, use subgroup means
        n_subgroups = len(data_arr) // subgroup_size
        subgroups = data_arr[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
        analysis_data = np.mean(subgroups, axis=1)
    elif chart_type == "I_MR":
        # Individual and moving range chart
        mean = np.mean(data_arr)
        mr = np.abs(np.diff(data_arr))
        mr_mean = np.mean(mr)
        d2 = 1.128  # For n=2
        sigma = mr_mean / d2

        control_limits = SPCControlLimits(
            center_line=mean,
            ucl=mean + 3 * sigma,
            lcl=mean - 3 * sigma,
            sigma=sigma
        )
        analysis_data = data_arr
    else:
        # Default: treat as individuals chart
        mean = np.mean(data_arr)
        sigma = np.std(data_arr, ddof=1)
        control_limits = SPCControlLimits(
            center_line=mean,
            ucl=mean + 3 * sigma,
            lcl=mean - 3 * sigma,
            sigma=sigma
        )
        analysis_data = data_arr

    # Check for violations
    violations = check_weco_rules(analysis_data, control_limits, rules)
    in_control = len(violations) == 0

    # Calculate capability indices if target is provided
    capability_indices = None
    if target is not None:
        sigma = control_limits.sigma
        # Simple Cp and Cpk calculation
        spec_range = 6 * sigma  # Assume ±3sigma as spec limits
        cp = spec_range / (6 * sigma) if sigma > 0 else 0.0
        cpk = min(
            (control_limits.ucl - target) / (3 * sigma),
            (target - control_limits.lcl) / (3 * sigma)
        ) if sigma > 0 else 0.0
        capability_indices = {"Cp": float(cp), "Cpk": float(cpk)}

    return SPCAnalysisResult(
        chart_type=chart_type,
        control_limits=control_limits,
        violations=violations,
        in_control=in_control,
        capability_indices=capability_indices
    )


def evaluate_math_expression(expression: str, variables: Dict[str, float], symbolic: bool = False) -> MathEvalResult:
    """Evaluate mathematical expression with given variables."""
    if symbolic:
        # Symbolic evaluation
        expr = sympify(expression)
        # Substitute variables
        for var, val in variables.items():
            expr = expr.subs(Symbol(var), val)
        result = float(expr.evalf())
    else:
        # Numerical evaluation
        # Create safe namespace
        safe_dict = {"__builtins__": {}}
        safe_dict.update(np.__dict__)  # Add numpy functions
        safe_dict.update(variables)

        try:
            result = eval(expression, safe_dict)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")

    return MathEvalResult(
        result=float(result),
        expression=expression,
        variables=variables
    )


def symbolic_derivative(expression: str, var: str) -> str:
    """Compute symbolic derivative."""
    x = Symbol(var)
    expr = sympify(expression)
    return str(diff(expr, x))
