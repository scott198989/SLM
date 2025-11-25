from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Factor:
    name: str
    levels: List[float]


@dataclass
class DesignDOE:
    type: str
    factors: List[Factor]
    response: str
    alpha: float = 0.05
    blocks: Optional[int] = None
    replicates: int = 1


@dataclass
class EvalSPC:
    chart: str
    subgroup_size: int
    data_source: str
    alpha: float = 0.05
    rules: List[str] = field(default_factory=lambda: ["WECO_1", "WECO_2", "WECO_3", "WECO_4"])


@dataclass
class MathExpression:
    expression: str
    variables: Dict[str, float] = field(default_factory=dict)
    symbolic: bool = False


@dataclass
class StatTest:
    test_type: str  # "ttest", "anova", "regression", "chi_square"
    data_a: Optional[List[float]] = None
    data_b: Optional[List[float]] = None
    formula: Optional[str] = None
    data_frame: Optional[Dict[str, List[Any]]] = None
    alpha: float = 0.05
    equal_var: bool = False


@dataclass
class DOEOperation:
    operation: str  # "factorial", "fractional_factorial", "response_surface", "box_behnken"
    factors: List[Factor]
    response_data: Optional[List[float]] = None
    center_points: int = 0
    blocks: Optional[int] = None


@dataclass
class SPCOperation:
    chart_type: str  # "XBar_R", "XBar_S", "I_MR", "P", "NP", "C", "U"
    data: List[float]
    subgroup_size: Optional[int] = None
    rules: List[str] = field(default_factory=lambda: ["WECO_1"])
    target: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None


@dataclass
class DSLRequest:
    design_doe: Optional[DesignDOE] = None
    eval_spc: Optional[EvalSPC] = None
    math_expr: Optional[MathExpression] = None
    stat_test: Optional[StatTest] = None
    doe_operation: Optional[DOEOperation] = None
    spc_operation: Optional[SPCOperation] = None
    raw: dict = field(default_factory=dict)
