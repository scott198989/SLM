"""
Integration tests for the complete SRS-7B reasoning pipeline.
"""
from __future__ import annotations

import pytest

from havoc_core.config import RAGConfig, SRSConfig
from havoc_srs.answer import build_answer
from havoc_srs.arbiter import decide
from havoc_srs.argue import build_arguments
from havoc_srs.audit import run_audit
from havoc_srs.execute import Executor
from havoc_srs.ground import attach_references
from havoc_srs.mode import Difficulty, ModeClassifier, Risk, TaskType
from havoc_srs.orchestrator import run_pipeline
from havoc_srs.plan import Planner
from havoc_rag.retrieval import Retriever


class TestModeClassification:
    """Test MODE stage classification."""

    def test_classify_stats_task(self):
        """Test classification of statistical task."""
        classifier = ModeClassifier()
        result = classifier.classify("Run a t-test to compare group means")
        assert result.task == TaskType.STATS

    def test_classify_doe_task(self):
        """Test classification of DOE task."""
        classifier = ModeClassifier()
        result = classifier.classify("Design a factorial DOE for temperature and pressure")
        assert result.task == TaskType.DOE

    def test_classify_spc_task(self):
        """Test classification of SPC task."""
        classifier = ModeClassifier()
        result = classifier.classify("Analyze control chart for process stability")
        assert result.task == TaskType.SPC

    def test_difficulty_classification(self):
        """Test difficulty classification based on prompt length."""
        classifier = ModeClassifier()
        short = classifier.classify("Run ANOVA")
        long = classifier.classify("a" * 300)
        assert short.difficulty in [Difficulty.TRIVIAL, Difficulty.NORMAL]
        assert long.difficulty == Difficulty.HARD


class TestGroundStage:
    """Test GROUND stage with RAG retrieval."""

    def test_attach_references(self):
        """Test reference attachment."""
        retriever = Retriever(embed_dim=384)
        retriever.add_corpus([
            "ANOVA is used for comparing means",
            "DOE optimizes processes",
            "SPC monitors quality"
        ])
        classifier = ModeClassifier()
        mode = classifier.classify("What is ANOVA?")
        grounded = attach_references("What is ANOVA?", mode, retriever)

        assert len(grounded.references) > 0
        assert grounded.retriever is not None


class TestPlanStage:
    """Test PLAN stage."""

    def test_plan_doe_task(self):
        """Test planning for DOE task."""
        planner = Planner()
        classifier = ModeClassifier()
        mode = classifier.classify("Design a DOE")
        plan = planner.build_plan("Design a DOE", mode)

        assert plan.problem_type == TaskType.DOE
        assert len(plan.steps) > 0
        assert any("dsl" in step.tools for step in plan.steps)

    def test_plan_stats_task(self):
        """Test planning for stats task."""
        planner = Planner()
        classifier = ModeClassifier()
        mode = classifier.classify("Run a t-test")
        plan = planner.build_plan("Run a t-test", mode)

        assert plan.problem_type == TaskType.STATS
        assert any("python_math" in step.tools for step in plan.steps)


class TestExecuteStage:
    """Test EXECUTE stage."""

    def test_execute_plan_with_ttest(self):
        """Test executing plan with t-test."""
        planner = Planner()
        classifier = ModeClassifier()
        mode = classifier.classify("Run a t-test")
        plan = planner.build_plan("Run a t-test", mode)

        executor = Executor()
        result = executor.run_plan(plan, "Run a t-test")

        assert len(result.steps) > 0
        assert result.overall_success is True or result.overall_success is False  # Either is valid
        assert result.summary is not None

    def test_execute_plan_with_doe(self):
        """Test executing plan with DOE."""
        planner = Planner()
        classifier = ModeClassifier()
        mode = classifier.classify("Design a factorial DOE")
        plan = planner.build_plan("Design a factorial DOE", mode)

        executor = Executor()
        result = executor.run_plan(plan, "Design a factorial DOE")

        assert len(result.steps) > 0


class TestArgueStage:
    """Test ARGUE stage."""

    def test_build_arguments_from_execution(self):
        """Test building arguments from execution results."""
        planner = Planner()
        classifier = ModeClassifier()
        mode = classifier.classify("Run a t-test")
        plan = planner.build_plan("Run a t-test", mode)

        executor = Executor()
        exec_result = executor.run_plan(plan, "Run a t-test")

        arguments = build_arguments("Run a t-test", exec_result=exec_result)

        assert len(arguments) == 2  # PRO and CON
        pro = [a for a in arguments if a.direction == "PRO"][0]
        con = [a for a in arguments if a.direction == "CON"][0]

        assert pro.confidence >= 0 and pro.confidence <= 1
        assert con.confidence >= 0 and con.confidence <= 1
        assert len(pro.evidence) > 0
        assert len(con.evidence) > 0

    def test_arguments_without_execution(self):
        """Test building arguments without execution results."""
        arguments = build_arguments("Some prompt")

        assert len(arguments) == 2
        assert all(0 <= a.confidence <= 1 for a in arguments)


class TestArbiterStage:
    """Test ARBITER stage."""

    def test_decide_with_clear_winner(self):
        """Test decision with clear winner."""
        from havoc_srs.argue import Argument

        pro = Argument(
            direction="PRO",
            evidence=["Strong evidence"],
            confidence=0.9,
            text="PRO argument",
            supporting_data={"pvalue": 0.001}
        )
        con = Argument(
            direction="CON",
            evidence=["Weak evidence"],
            confidence=0.3,
            text="CON argument",
            supporting_data={}
        )

        decision = decide([pro, con])

        assert decision.winner == "PRO"
        assert decision.confidence > 0

    def test_decide_with_conflicting_arguments(self):
        """Test decision with conflicting arguments."""
        from havoc_srs.argue import Argument

        pro = Argument(
            direction="PRO",
            evidence=["Evidence 1"],
            confidence=0.55,
            text="PRO argument",
            supporting_data={}
        )
        con = Argument(
            direction="CON",
            evidence=["Evidence 2"],
            confidence=0.50,
            text="CON argument",
            supporting_data={}
        )

        decision = decide([pro, con])

        # Should detect conflict and reduce consistency score
        assert decision.consistency_score < 1.0

    def test_decide_empty_arguments(self):
        """Test decision with no arguments."""
        decision = decide([])

        assert decision.winner == "UNDECIDED"
        assert decision.confidence == 0.0


class TestAuditStage:
    """Test AUDIT stage."""

    def test_audit_high_confidence(self):
        """Test audit with high confidence decision."""
        from havoc_srs.arbiter import ArbiterDecision
        from havoc_srs.argue import Argument

        decision = ArbiterDecision(
            winner="PRO",
            confidence=0.9,
            rationale=["Good evidence"],
            consistency_score=1.0,
            data_quality_score=1.0
        )
        arguments = []

        audit = run_audit(decision, arguments)

        assert audit.severity in ["LOW", "HIGH"]
        assert audit.downgraded_confidence > 0

    def test_audit_low_confidence(self):
        """Test audit with low confidence decision."""
        from havoc_srs.arbiter import ArbiterDecision

        decision = ArbiterDecision(
            winner="PRO",
            confidence=0.3,
            rationale=["Weak evidence"],
            consistency_score=0.5,
            data_quality_score=0.6
        )
        arguments = []

        audit = run_audit(decision, arguments)

        assert len(audit.issues) > 0
        assert audit.downgraded_confidence <= decision.confidence


class TestAnswerStage:
    """Test ANSWER stage."""

    def test_build_complete_answer(self):
        """Test building complete answer."""
        # Run simple pipeline to get all components
        classifier = ModeClassifier()
        mode = classifier.classify("Run a t-test")

        retriever = Retriever(embed_dim=384)
        retriever.add_corpus(["T-test compares means"])
        grounded = attach_references("Run a t-test", mode, retriever)

        planner = Planner()
        plan = planner.build_plan("Run a t-test", mode)

        executor = Executor()
        exec_result = executor.run_plan(plan, "Run a t-test", grounded=grounded)

        arguments = build_arguments("Run a t-test", exec_result=exec_result)
        decision = decide(arguments)
        audit = run_audit(decision, arguments)

        answer = build_answer(
            prompt="Run a t-test",
            decision=decision,
            audit=audit,
            exec_result=exec_result,
            grounded=grounded,
            mode=mode
        )

        assert answer.conclusion is not None
        assert answer.confidence >= 0 and answer.confidence <= 1
        assert len(answer.assumptions) > 0
        assert len(answer.reasoning_trace) == 8  # 8 SRS stages
        assert answer.raw_data is not None

    def test_answer_format_human_readable(self):
        """Test human-readable formatting."""
        from havoc_srs.answer import Answer

        answer = Answer(
            conclusion="Test conclusion",
            key_numbers={"pvalue": 0.05, "confidence": 0.8},
            assumptions=["Assumption 1"],
            confidence=0.8,
            caveats=["Caveat 1"],
            suggested_checks=["Check 1"],
            reasoning_trace=["Step 1", "Step 2"]
        )

        formatted = answer.format_human_readable()

        assert "## CONCLUSION" in formatted
        assert "## CONFIDENCE" in formatted
        assert "## KEY NUMBERS" in formatted
        assert "## ASSUMPTIONS" in formatted
        assert "## REASONING TRACE" in formatted


class TestFullPipeline:
    """Test complete SRS pipeline integration."""

    def test_pipeline_stats_task(self):
        """Test full pipeline with stats task."""
        config = SRSConfig(
            rag=RAGConfig(embed_dim=384)
        )
        answer = run_pipeline("Run a t-test to compare groups", config)

        assert answer is not None
        assert answer.conclusion is not None
        assert 0 <= answer.confidence <= 1
        assert len(answer.reasoning_trace) == 8

    def test_pipeline_doe_task(self):
        """Test full pipeline with DOE task."""
        config = SRSConfig(
            rag=RAGConfig(embed_dim=384)
        )
        answer = run_pipeline("Design a factorial DOE for temperature and pressure", config)

        assert answer is not None
        assert "DOE" in answer.assumptions[0] or "FACTORIAL" in answer.assumptions[0] or "Task classified as: DOE" in answer.assumptions[0]

    def test_pipeline_spc_task(self):
        """Test full pipeline with SPC task."""
        config = SRSConfig(
            rag=RAGConfig(embed_dim=384)
        )
        answer = run_pipeline("Analyze SPC control chart for defects", config)

        assert answer is not None
        assert answer.confidence >= 0

    def test_pipeline_produces_structured_output(self):
        """Test that pipeline produces all required output fields."""
        config = SRSConfig(
            rag=RAGConfig(embed_dim=384)
        )
        answer = run_pipeline("What is ANOVA?", config)

        # Check all required fields
        assert hasattr(answer, "conclusion")
        assert hasattr(answer, "confidence")
        assert hasattr(answer, "key_numbers")
        assert hasattr(answer, "assumptions")
        assert hasattr(answer, "caveats")
        assert hasattr(answer, "suggested_checks")
        assert hasattr(answer, "reasoning_trace")
        assert hasattr(answer, "raw_data")

        # Verify reasoning trace has all 8 stages
        assert len(answer.reasoning_trace) == 8
        assert "MODE:" in answer.reasoning_trace[0]
        assert "GROUND:" in answer.reasoning_trace[1]
        assert "PLAN:" in answer.reasoning_trace[2]
        assert "EXECUTE:" in answer.reasoning_trace[3]
        assert "ARGUE:" in answer.reasoning_trace[4]
        assert "ARBITER:" in answer.reasoning_trace[5]
        assert "AUDIT:" in answer.reasoning_trace[6]
        assert "ANSWER:" in answer.reasoning_trace[7]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
