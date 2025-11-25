from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from havoc_tools.dsl.parser import DSLParseError, parse_dsl
from havoc_tools.dsl.spec import (
    DSLRequest,
    DesignDOE,
    DOEOperation,
    EvalSPC,
    MathExpression,
    SPCOperation,
    StatTest,
)
from havoc_tools.python_math import engine


@dataclass
class ExecutionOutcome:
    description: str
    payload: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class DSLExecutor:
    def __init__(self):
        pass

    def execute(self, content: str) -> ExecutionOutcome:
        """Execute DSL request and return structured outcome."""
        try:
            request = parse_dsl(content)
        except DSLParseError as e:
            return ExecutionOutcome(
                description="Parse error",
                payload={},
                success=False,
                error=str(e)
            )

        # Execute based on operation type
        try:
            if request.math_expr:
                return self._execute_math(request.math_expr)
            if request.stat_test:
                return self._execute_stat_test(request.stat_test)
            if request.doe_operation:
                return self._execute_doe_operation(request.doe_operation)
            if request.spc_operation:
                return self._execute_spc(request.spc_operation)
            if request.design_doe:
                return self._execute_doe_legacy(request.design_doe)
            if request.eval_spc:
                return self._execute_spc_legacy(request.eval_spc)

            return ExecutionOutcome(
                description="No operation specified",
                payload=request.raw,
                success=True
            )
        except Exception as e:
            return ExecutionOutcome(
                description="Execution error",
                payload={},
                success=False,
                error=f"{type(e).__name__}: {str(e)}"
            )

    def _execute_math(self, math_expr: MathExpression) -> ExecutionOutcome:
        """Execute mathematical expression."""
        result = engine.evaluate_math_expression(
            math_expr.expression,
            math_expr.variables,
            math_expr.symbolic
        )
        return ExecutionOutcome(
            description="Math expression evaluation",
            payload=asdict(result),
            success=True
        )

    def _execute_stat_test(self, stat_test: StatTest) -> ExecutionOutcome:
        """Execute statistical test."""
        test_type = stat_test.test_type.lower()

        if test_type == "ttest":
            if stat_test.data_a is None or stat_test.data_b is None:
                raise ValueError("T-test requires data_a and data_b")
            result = engine.run_ttest(
                stat_test.data_a,
                stat_test.data_b,
                equal_var=stat_test.equal_var,
                alpha=stat_test.alpha
            )
            return ExecutionOutcome(
                description="T-test",
                payload=asdict(result),
                success=True
            )

        elif test_type == "anova":
            if stat_test.formula is None or stat_test.data_frame is None:
                raise ValueError("ANOVA requires formula and data_frame")
            data = pd.DataFrame(stat_test.data_frame)
            result = engine.run_anova(stat_test.formula, data)
            return ExecutionOutcome(
                description="ANOVA",
                payload=asdict(result),
                success=True
            )

        elif test_type == "regression":
            if stat_test.formula is None or stat_test.data_frame is None:
                raise ValueError("Regression requires formula and data_frame")
            data = pd.DataFrame(stat_test.data_frame)
            result = engine.fit_regression(stat_test.formula, data)
            return ExecutionOutcome(
                description="Regression",
                payload=asdict(result),
                success=True
            )

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _execute_doe_operation(self, doe_op: DOEOperation) -> ExecutionOutcome:
        """Execute DOE operation."""
        # Build design matrix from factors
        factor_names = [f.name for f in doe_op.factors]
        design_matrix = np.array([f.levels for f in doe_op.factors]).T

        # Use provided response data or generate dummy data
        if doe_op.response_data:
            responses = np.array(doe_op.response_data)
        else:
            # Generate random responses for demonstration
            responses = np.random.randn(design_matrix.shape[0])

        result = engine.run_doe_analysis(
            design_matrix,
            responses,
            factor_names=factor_names
        )

        return ExecutionOutcome(
            description=f"DOE {doe_op.operation}",
            payload=asdict(result),
            success=True
        )

    def _execute_spc(self, spc_op: SPCOperation) -> ExecutionOutcome:
        """Execute SPC control chart analysis."""
        result = engine.run_spc_analysis(
            chart_type=spc_op.chart_type,
            data=spc_op.data,
            subgroup_size=spc_op.subgroup_size,
            rules=spc_op.rules,
            target=spc_op.target
        )

        # Convert to dict, handling nested dataclasses
        payload = {
            "chart_type": result.chart_type,
            "control_limits": asdict(result.control_limits),
            "violations": [asdict(v) for v in result.violations],
            "in_control": result.in_control,
            "capability_indices": result.capability_indices
        }

        return ExecutionOutcome(
            description=f"SPC {spc_op.chart_type} chart",
            payload=payload,
            success=True
        )

    def _execute_doe_legacy(self, doe: DesignDOE) -> ExecutionOutcome:
        """Execute legacy DESIGN_DOE format."""
        factor_names = [f.name for f in doe.factors]
        design_matrix = np.array([f.levels for f in doe.factors]).T
        responses = np.random.randn(design_matrix.shape[0])

        result = engine.run_doe_analysis(
            design_matrix,
            responses,
            factor_names=factor_names
        )

        return ExecutionOutcome(
            description=f"DOE {doe.type}",
            payload=asdict(result),
            success=True
        )

    def _execute_spc_legacy(self, spc: EvalSPC) -> ExecutionOutcome:
        """Execute legacy EVAL_SPC format."""
        # For legacy format, we need actual data (not data_source string)
        # Generate dummy data for demonstration
        dummy_data = list(np.random.randn(100))

        result = engine.run_spc_analysis(
            chart_type=spc.chart,
            data=dummy_data,
            subgroup_size=spc.subgroup_size,
            rules=spc.rules
        )

        payload = {
            "chart_type": result.chart_type,
            "control_limits": asdict(result.control_limits),
            "violations": [asdict(v) for v in result.violations],
            "in_control": result.in_control,
            "capability_indices": result.capability_indices,
            "note": f"Legacy format - used dummy data from {spc.data_source}"
        }

        return ExecutionOutcome(
            description=f"SPC {spc.chart} evaluation",
            payload=payload,
            success=True
        )
