from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sympy import Symbol, diff


@dataclass
class TTestResult:
    statistic: float
    pvalue: float
    df: float


@dataclass
class ANOVAResult:
    table: Dict[str, Any]


@dataclass
class RegressionResult:
    params: Dict[str, float]
    pvalues: Dict[str, float]
    r_squared: float


@dataclass
class DOEAnalysisResult:
    effects: Dict[str, float]
    summary: str


def run_ttest(sample_a: List[float], sample_b: List[float], equal_var: bool = False) -> TTestResult:
    stat, pval = stats.ttest_ind(sample_a, sample_b, equal_var=equal_var)
    df = len(sample_a) + len(sample_b) - 2
    return TTestResult(statistic=float(stat), pvalue=float(pval), df=df)


def run_anova(formula: str, data) -> ANOVAResult:
    model = smf.ols(formula=formula, data=data).fit()
    table = sm.stats.anova_lm(model, typ=2).to_dict()
    return ANOVAResult(table=table)


def fit_regression(formula: str, data) -> RegressionResult:
    model = smf.ols(formula=formula, data=data).fit()
    return RegressionResult(params=dict(model.params), pvalues=dict(model.pvalues), r_squared=float(model.rsquared))


def run_doe_analysis(design_matrix: np.ndarray, responses: np.ndarray) -> DOEAnalysisResult:
    X = sm.add_constant(design_matrix)
    model = sm.OLS(responses, X).fit()
    effects = {f"beta_{i}": float(val) for i, val in enumerate(model.params)}
    summary = model.summary().as_text()
    return DOEAnalysisResult(effects=effects, summary=summary)


def symbolic_derivative(expression: str, var: str) -> str:
    x = Symbol(var)
    return str(diff(expression, x))
