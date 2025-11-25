#!/usr/bin/env python
"""
Smoke test for SRS-7B and DSL implementation.
Tests basic functionality without requiring pytest.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/user/SLM/src")

from havoc_core.config import RAGConfig, SRSConfig
from havoc_srs.orchestrator import run_pipeline
from havoc_tools.dsl.executor import DSLExecutor
from havoc_tools.python_math import engine


def test_dsl_math():
    """Test DSL math expression execution."""
    print("Testing DSL math expression...")
    executor = DSLExecutor()
    dsl = json.dumps({
        "MATH": {
            "expression": "x**2 + y",
            "variables": {"x": 2, "y": 3}
        }
    })
    result = executor.execute(dsl)
    assert result.success, f"Math execution failed: {result.error}"
    assert result.payload["result"] == 7.0, f"Expected 7.0, got {result.payload['result']}"
    print("✓ DSL math expression test passed")


def test_dsl_ttest():
    """Test DSL t-test execution."""
    print("\nTesting DSL t-test...")
    executor = DSLExecutor()
    dsl = json.dumps({
        "STAT_TEST": {
            "test_type": "ttest",
            "data_a": [1, 2, 3, 4, 5],
            "data_b": [2, 3, 4, 5, 6]
        }
    })
    result = executor.execute(dsl)
    assert result.success, f"T-test execution failed: {result.error}"
    assert "pvalue" in result.payload, "T-test result missing pvalue"
    assert "statistic" in result.payload, "T-test result missing statistic"
    print(f"✓ DSL t-test passed (p={result.payload['pvalue']:.4f})")


def test_dsl_spc():
    """Test DSL SPC execution."""
    print("\nTesting DSL SPC analysis...")
    executor = DSLExecutor()
    data = [10.0] * 15 + [20.0]  # Outlier at end
    dsl = json.dumps({
        "SPC": {
            "chart_type": "I_MR",
            "data": data,
            "rules": ["WECO_1"]
        }
    })
    result = executor.execute(dsl)
    assert result.success, f"SPC execution failed: {result.error}"
    assert "control_limits" in result.payload, "SPC result missing control_limits"
    assert "in_control" in result.payload, "SPC result missing in_control status"
    print(f"✓ DSL SPC passed (in_control={result.payload['in_control']})")


def test_engine_ttest():
    """Test engine t-test function."""
    print("\nTesting engine t-test...")
    result = engine.run_ttest([1, 2, 3], [2, 3, 4])
    assert result.pvalue > 0, "T-test pvalue should be positive"
    assert result.df == 4, f"Expected df=4, got {result.df}"
    assert hasattr(result, "ci_low"), "Missing confidence interval"
    assert hasattr(result, "ci_high"), "Missing confidence interval"
    print(f"✓ Engine t-test passed (p={result.pvalue:.4f}, CI=[{result.ci_low:.2f}, {result.ci_high:.2f}])")


def test_engine_spc():
    """Test engine SPC analysis."""
    print("\nTesting engine SPC...")
    data = [10.0] * 10 + [20.0] + [10.0] * 10
    result = engine.run_spc_analysis(
        chart_type="I_MR",
        data=data,
        rules=["WECO_1"]
    )
    assert result.control_limits.center_line > 0, "Control limits not calculated"
    assert result.in_control is False, "Should detect outlier"
    assert len(result.violations) > 0, "Should detect violations"
    print(f"✓ Engine SPC passed (violations={len(result.violations)})")


def test_srs_pipeline():
    """Test complete SRS pipeline."""
    print("\nTesting complete SRS pipeline...")
    config = SRSConfig(rag=RAGConfig(embed_dim=384))
    answer = run_pipeline("Run a t-test to compare two groups", config)

    assert answer is not None, "Pipeline returned None"
    assert answer.conclusion is not None, "Answer missing conclusion"
    assert 0 <= answer.confidence <= 1, f"Invalid confidence: {answer.confidence}"
    assert len(answer.reasoning_trace) == 8, f"Expected 8 reasoning steps, got {len(answer.reasoning_trace)}"

    # Check all 8 stages are in reasoning trace
    stages = ["MODE:", "GROUND:", "PLAN:", "EXECUTE:", "ARGUE:", "ARBITER:", "AUDIT:", "ANSWER:"]
    for i, stage in enumerate(stages):
        assert stage in answer.reasoning_trace[i], f"Missing {stage} in reasoning trace"

    print(f"✓ SRS pipeline passed (confidence={answer.confidence:.2f})")
    print("\nReasoning trace:")
    for i, step in enumerate(answer.reasoning_trace, 1):
        print(f"  {i}. {step}")


def test_answer_formatting():
    """Test answer human-readable formatting."""
    print("\nTesting answer formatting...")
    config = SRSConfig(rag=RAGConfig(embed_dim=384))
    answer = run_pipeline("Design a DOE for temperature and pressure", config)

    formatted = answer.format_human_readable()
    assert "## CONCLUSION" in formatted, "Missing CONCLUSION section"
    assert "## CONFIDENCE" in formatted, "Missing CONFIDENCE section"
    assert "## ASSUMPTIONS" in formatted, "Missing ASSUMPTIONS section"
    assert "## REASONING TRACE" in formatted, "Missing REASONING TRACE section"

    print("✓ Answer formatting passed")
    print("\n" + "="*60)
    print("FORMATTED OUTPUT:")
    print("="*60)
    print(formatted)


def main():
    """Run all smoke tests."""
    print("="*60)
    print("SRS-7B and DSL Smoke Tests")
    print("="*60)

    tests = [
        test_dsl_math,
        test_dsl_ttest,
        test_dsl_spc,
        test_engine_ttest,
        test_engine_spc,
        test_srs_pipeline,
        test_answer_formatting,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    if failed == 0:
        print(f"All {len(tests)} tests PASSED ✓")
        print("="*60)
        return 0
    else:
        print(f"{failed}/{len(tests)} tests FAILED ✗")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
