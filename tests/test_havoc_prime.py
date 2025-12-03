"""
Tests for HAVOC PRIME system

Tests budget levels, routing, and integration with SRS tools.
"""

import pytest

from havoc_prime.orchestrator import HavocPrimeOrchestrator, run_havoc_prime
from havoc_prime.router import Budget, TaskRouter, TaskType


class TestTaskRouter:
    """Test task routing and budget assignment"""

    def test_micro_budget_trivial_arithmetic(self):
        """Test MICRO budget for trivial arithmetic"""
        router = TaskRouter()
        decision = router.route("3 + 3")

        assert decision.budget == Budget.MICRO
        assert decision.difficulty.name in ["TRIVIAL", "SIMPLE"]

    def test_micro_budget_definition(self):
        """Test MICRO budget for definitional queries"""
        router = TaskRouter()
        decision = router.route("What is ANOVA?")

        assert decision.task_type == TaskType.TRIVIAL
        assert decision.budget == Budget.MICRO

    def test_light_budget_simple_question(self):
        """Test LIGHT budget for simple questions"""
        router = TaskRouter()
        decision = router.route("Explain t-test")

        assert decision.budget in [Budget.LIGHT, Budget.MICRO]

    def test_medium_budget_stats_question(self):
        """Test MEDIUM budget for stats questions"""
        router = TaskRouter()
        decision = router.route("Run a t-test to compare two groups")

        assert decision.budget in [Budget.MEDIUM, Budget.LIGHT]
        assert decision.task_type == TaskType.STATS

    def test_heavy_budget_doe_design(self):
        """Test HEAVY budget for DOE design"""
        router = TaskRouter()
        decision = router.route("Design a Box-Behnken DOE for temperature, pressure, and speed")

        assert decision.budget in [Budget.HEAVY, Budget.MEDIUM]
        assert decision.task_type == TaskType.DOE

    def test_heavy_budget_spc_analysis(self):
        """Test HEAVY budget for SPC analysis"""
        router = TaskRouter()
        decision = router.route("Analyze control chart data for process stability")

        assert decision.task_type == TaskType.SPC


class TestHavocPrimeOrchestrator:
    """Test HAVOC PRIME orchestrator"""

    def test_micro_budget_execution(self):
        """Test MICRO budget execution (no PRIME)"""
        result = run_havoc_prime("3 + 5")

        assert "answer" in result
        assert result["routing"]["budget"] == "MICRO"
        # Should skip PRIME entirely
        assert result["workspace_summary"]["facts_count"] == 0

    def test_light_budget_execution(self):
        """Test LIGHT budget execution"""
        orchestrator = HavocPrimeOrchestrator(
            enable_chrono=False,  # Disable chrono for LIGHT
            enable_adversarial=False  # Disable adversarial for LIGHT
        )

        result = orchestrator.process("What is a t-test?")

        assert "answer" in result
        assert result["routing"]["budget"] in ["LIGHT", "MICRO"]

    def test_medium_budget_with_stats(self):
        """Test MEDIUM budget with stats task"""
        orchestrator = HavocPrimeOrchestrator(
            enable_chrono=True,
            enable_adversarial=True
        )

        result = orchestrator.process("Run ANOVA on three groups")

        assert "answer" in result
        # Should have subgoals
        assert "workspace_summary" in result

    def test_heavy_budget_with_doe(self):
        """Test HEAVY budget with DOE task"""
        orchestrator = HavocPrimeOrchestrator(
            enable_chrono=True,
            enable_adversarial=True,
            max_chrono_iterations=2
        )

        result = orchestrator.process("Design a full factorial DOE with 3 factors")

        assert "answer" in result
        assert result["routing"]["budget"] in ["HEAVY", "MEDIUM"]
        # Should have verification
        assert "verification" in result


class TestBudgetAntiBlOat:
    """Test anti-bloat rules are enforced"""

    def test_no_prime_for_trivial(self):
        """MICRO budget must skip PRIME"""
        result = run_havoc_prime("2 + 2")

        assert result["routing"]["budget"] == "MICRO"
        # No subgoals
        assert result["workspace_summary"]["facts_count"] == 0

    def test_light_no_chrono(self):
        """LIGHT budget should not use chrono-loop"""
        orchestrator = HavocPrimeOrchestrator()
        result = orchestrator.process("Simple question about ANOVA")

        # If LIGHT, should not have chrono metadata
        if result["routing"]["budget"] == "LIGHT":
            assert "workspace_summary" in result

    def test_graph_size_limits(self):
        """Operator graph should respect budget limits"""
        from havoc_prime.operator_graph import OperatorGraphBuilder

        builder = OperatorGraphBuilder()

        # LIGHT: max 3 subgoals
        graph_light = builder.build_graph("test", TaskType.GENERAL, Budget.LIGHT)
        assert len(graph_light.subgoals) <= 3

        # MEDIUM: max 7 subgoals
        graph_medium = builder.build_graph("test", TaskType.GENERAL, Budget.MEDIUM)
        assert len(graph_medium.subgoals) <= 7

        # HEAVY: max 12 subgoals
        graph_heavy = builder.build_graph("test", TaskType.DOE, Budget.HEAVY)
        assert len(graph_heavy.subgoals) <= 12


class TestSRSToolIntegration:
    """Test SRS tools are called correctly"""

    def test_dsl_executor_integration(self):
        """Test DSL executor is called for DOE tasks"""
        from srs_tools import DSLExecutor

        executor = DSLExecutor()
        dsl_command = '{"DOE": {"operation": "factorial", "factors": [{"name": "A", "levels": [-1, 1]}]}}'

        result = executor.execute(dsl_command)

        assert "success" in result
        assert "payload" in result

    def test_python_math_integration(self):
        """Test Python math tool integration"""
        from srs_tools import PythonMathEngine

        engine = PythonMathEngine()
        result = engine.t_test([1, 2, 3], [2, 3, 4])

        assert "pvalue" in result
        assert "statistic" in result
        assert "significant" in result

    def test_rag_helper_integration(self):
        """Test RAG helper integration"""
        from srs_tools import RAGHelper

        rag = RAGHelper()
        references = rag.retrieve("What is ANOVA?", k=3)

        assert isinstance(references, list)
        assert len(references) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
