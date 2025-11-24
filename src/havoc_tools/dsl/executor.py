from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from havoc_tools.dsl.parser import parse_dsl
from havoc_tools.dsl.spec import DSLRequest, DesignDOE
from havoc_tools.python_math import engine


@dataclass
class ExecutionOutcome:
    description: str
    payload: dict


class DSLExecutor:
    def __init__(self):
        pass

    def execute(self, content: str) -> ExecutionOutcome:
        request = parse_dsl(content)
        if request.design_doe:
            return self._execute_doe(request.design_doe)
        if request.eval_spc:
            return ExecutionOutcome(description="SPC evaluation stub", payload=request.raw)
        return ExecutionOutcome(description="No-op", payload={})

    def _execute_doe(self, doe: DesignDOE) -> ExecutionOutcome:
        design_matrix = np.array([factor.levels for factor in doe.factors]).T
        responses = np.random.randn(design_matrix.shape[0])
        doe_result = engine.run_doe_analysis(design_matrix, responses)
        return ExecutionOutcome(description=f"DOE {doe.type}", payload=doe_result.__dict__)
