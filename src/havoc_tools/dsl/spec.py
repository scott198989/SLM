from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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


@dataclass
class EvalSPC:
    chart: str
    subgroup_size: int
    data_source: str
    alpha: float = 0.05


@dataclass
class DSLRequest:
    design_doe: Optional[DesignDOE] = None
    eval_spc: Optional[EvalSPC] = None
    raw: dict = field(default_factory=dict)
