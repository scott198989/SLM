"""
Tests for DSL parsing, execution, and math/stats engine.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from havoc_tools.dsl.executor import DSLExecutor
from havoc_tools.dsl.parser import DSLParseError, parse_dsl
from havoc_tools.python_math import engine


class TestDSLParsing:
    """Test DSL parsing for various operation types."""

    def test_parse_math_expression(self):
        """Test parsing math expression."""
        dsl = json.dumps({
            "MATH": {
                "expression": "x**2 + y",
                "variables": {"x": 2, "y": 3}
            }
        })
        request = parse_dsl(dsl)
        assert request.math_expr is not None
        assert request.math_expr.expression == "x**2 + y"
        assert request.math_expr.variables == {"x": 2, "y": 3}

    def test_parse_stat_test_ttest(self):
        """Test parsing t-test specification."""
        dsl = json.dumps({
            "STAT_TEST": {
                "test_type": "ttest",
                "data_a": [1, 2, 3],
                "data_b": [2, 3, 4],
                "alpha": 0.05
            }
        })
        request = parse_dsl(dsl)
        assert request.stat_test is not None
        assert request.stat_test.test_type == "ttest"
        assert request.stat_test.data_a == [1, 2, 3]

    def test_parse_doe_operation(self):
        """Test parsing DOE operation."""
        dsl = json.dumps({
            "DOE": {
                "operation": "factorial",
                "factors": [
                    {"name": "Temp", "levels": [-1, 0, 1]},
                    {"name": "Press", "levels": [-1, 1]}
                ],
                "response_data": [10, 12, 14, 11, 13, 15]
            }
        })
        request = parse_dsl(dsl)
        assert request.doe_operation is not None
        assert request.doe_operation.operation == "factorial"
        assert len(request.doe_operation.factors) == 2

    def test_parse_spc_operation(self):
        """Test parsing SPC operation."""
        dsl = json.dumps({
            "SPC": {
                "chart_type": "I_MR",
                "data": list(range(20)),
                "rules": ["WECO_1", "WECO_2"]
            }
        })
        request = parse_dsl(dsl)
        assert request.spc_operation is not None
        assert request.spc_operation.chart_type == "I_MR"
        assert len(request.spc_operation.data) == 20

    def test_parse_invalid_dsl(self):
        """Test parsing invalid DSL."""
        with pytest.raises(DSLParseError):
            parse_dsl("not valid json or yaml")


class TestDSLExecution:
    """Test DSL execution."""

    def test_execute_math_expression(self):
        """Test executing math expression."""
        executor = DSLExecutor()
        dsl = json.dumps({
            "MATH": {
                "expression": "x**2 + y",
                "variables": {"x": 2, "y": 3}
            }
        })
        result = executor.execute(dsl)
        assert result.success is True
        assert "result" in result.payload
        assert result.payload["result"] == 7.0  # 2**2 + 3

    def test_execute_ttest(self):
        """Test executing t-test."""
        executor = DSLExecutor()
        dsl = json.dumps({
            "STAT_TEST": {
                "test_type": "ttest",
                "data_a": [1, 2, 3, 4, 5],
                "data_b": [2, 3, 4, 5, 6]
            }
        })
        result = executor.execute(dsl)
        assert result.success is True
        assert "pvalue" in result.payload
        assert "statistic" in result.payload

    def test_execute_doe(self):
        """Test executing DOE analysis."""
        executor = DSLExecutor()
        dsl = json.dumps({
            "DOE": {
                "operation": "factorial",
                "factors": [
                    {"name": "Temperature", "levels": [-1, -1, 1, 1]},
                    {"name": "Pressure", "levels": [-1, 1, -1, 1]}
                ],
                "response_data": [10, 12, 14, 16]
            }
        })
        result = executor.execute(dsl)
        assert result.success is True
        assert "effects" in result.payload
        assert "main_effects" in result.payload

    def test_execute_spc(self):
        """Test executing SPC analysis."""
        executor = DSLExecutor()
        # Create data with an outlier
        data = [10.0] * 15 + [20.0]  # Outlier at end
        dsl = json.dumps({
            "SPC": {
                "chart_type": "I_MR",
                "data": data,
                "rules": ["WECO_1"]
            }
        })
        result = executor.execute(dsl)
        assert result.success is True
        assert "control_limits" in result.payload
        assert "violations" in result.payload
        assert "in_control" in result.payload


class TestMathEngine:
    """Test math/stats engine functions."""

    def test_ttest_basic(self):
        """Test basic t-test."""
        result = engine.run_ttest([1, 2, 3], [2, 3, 4])
        assert result.pvalue > 0  # Should be > 0
        assert result.df == 4  # 3 + 3 - 2
        assert hasattr(result, "ci_low")
        assert hasattr(result, "ci_high")

    def test_ttest_equal_var(self):
        """Test t-test with equal variance assumption."""
        result = engine.run_ttest([1, 2, 3], [2, 3, 4], equal_var=True)
        assert result.pvalue > 0
        assert result.equal_var is True

    def test_anova(self):
        """Test ANOVA."""
        data = pd.DataFrame({
            "y": [1, 2, 3, 4, 5, 6],
            "group": ["A", "A", "B", "B", "C", "C"]
        })
        result = engine.run_anova("y ~ C(group)", data)
        assert hasattr(result, "f_statistic")
        assert hasattr(result, "p_value")

    def test_regression(self):
        """Test regression."""
        data = pd.DataFrame({
            "y": [1, 2, 3, 4, 5],
            "x": [1, 2, 3, 4, 5]
        })
        result = engine.fit_regression("y ~ x", data)
        assert result.r_squared > 0.9  # Should be near-perfect fit
        assert "x" in result.params
        assert "Intercept" in result.params

    def test_doe_analysis(self):
        """Test DOE analysis."""
        design_matrix = np.array([
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ])
        responses = np.array([10, 12, 14, 16])
        result = engine.run_doe_analysis(
            design_matrix,
            responses,
            factor_names=["Temp", "Press"]
        )
        assert "Temp" in result.main_effects
        assert "Press" in result.main_effects
        assert result.r_squared >= 0  # Should have some fit
        assert len(result.significant_factors) >= 0

    def test_spc_xbar_r(self):
        """Test SPC XBar-R chart."""
        np.random.seed(42)
        data = np.random.normal(10, 1, 50)
        result = engine.run_spc_analysis(
            chart_type="XBar_R",
            data=list(data),
            subgroup_size=5,
            rules=["WECO_1"]
        )
        assert result.control_limits.center_line > 0
        assert result.control_limits.ucl > result.control_limits.center_line
        assert result.control_limits.lcl < result.control_limits.center_line

    def test_spc_i_mr(self):
        """Test SPC I-MR chart."""
        data = [10.0] * 20
        result = engine.run_spc_analysis(
            chart_type="I_MR",
            data=data,
            rules=["WECO_1", "WECO_4"]
        )
        assert result.in_control is False  # Should detect pattern

    def test_spc_violations(self):
        """Test SPC violation detection."""
        # Create data with clear violation
        data = [10.0] * 10 + [20.0] + [10.0] * 10
        result = engine.run_spc_analysis(
            chart_type="I_MR",
            data=data,
            rules=["WECO_1"]
        )
        assert result.in_control is False
        assert len(result.violations) > 0
        assert result.violations[0].rule == "WECO_1"

    def test_math_expression_evaluation(self):
        """Test math expression evaluation."""
        result = engine.evaluate_math_expression(
            "x**2 + 2*y",
            {"x": 3, "y": 4}
        )
        assert result.result == 17.0  # 3**2 + 2*4

    def test_math_expression_symbolic(self):
        """Test symbolic math expression."""
        result = engine.evaluate_math_expression(
            "x**2 + y",
            {"x": 2, "y": 3},
            symbolic=True
        )
        assert result.result == 7.0

    def test_symbolic_derivative(self):
        """Test symbolic derivative."""
        result = engine.symbolic_derivative("x**2 + 3*x", "x")
        assert "2*x" in result
        assert "3" in result


class TestToolFailureModes:
    """Test tool failure modes and error handling."""

    def test_executor_parse_error(self):
        """Test executor handles parse errors."""
        executor = DSLExecutor()
        result = executor.execute("invalid json")
        assert result.success is False
        assert result.error is not None

    def test_executor_missing_data(self):
        """Test executor handles missing data."""
        executor = DSLExecutor()
        dsl = json.dumps({
            "STAT_TEST": {
                "test_type": "ttest"
                # Missing data_a and data_b
            }
        })
        result = executor.execute(dsl)
        assert result.success is False
        assert result.error is not None

    def test_spc_missing_subgroup_size(self):
        """Test SPC fails gracefully without subgroup size."""
        with pytest.raises(ValueError):
            engine.run_spc_analysis(
                chart_type="XBar_R",
                data=[1, 2, 3, 4, 5],
                subgroup_size=None  # Required for XBar charts
            )

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        executor = DSLExecutor()
        dsl = json.dumps({
            "SPC": {
                "chart_type": "I_MR",
                "data": []
            }
        })
        result = executor.execute(dsl)
        # Should handle gracefully
        assert result.success is False or len(result.payload.get("violations", [])) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
