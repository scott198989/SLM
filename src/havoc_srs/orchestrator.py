from __future__ import annotations

from havoc_core.config import SRSConfig
from havoc_rag.retrieval import Retriever
from havoc_srs.answer import Answer, build_answer
from havoc_srs.argue import build_arguments
from havoc_srs.arbiter import decide
from havoc_srs.audit import run_audit
from havoc_srs.execute import Executor
from havoc_srs.ground import GroundedContext, attach_references
from havoc_srs.mode import ModeClassifier
from havoc_srs.plan import Planner


def run_pipeline(prompt: str, config: SRSConfig) -> Answer:
    """
    Run complete SRS-7B reasoning pipeline.

    Pipeline stages:
    1. MODE: Classify the task type, difficulty, and risk
    2. GROUND: Retrieve relevant references from RAG
    3. PLAN: Create execution plan based on task
    4. EXECUTE: Run plan steps using DSL and tools
    5. ARGUE: Build PRO/CON arguments from execution results
    6. ARBITER: Decide between arguments with consistency checks
    7. AUDIT: Review decision and downgrade confidence if needed
    8. ANSWER: Build human-readable structured answer

    Args:
        prompt: User's question or task
        config: SRS configuration

    Returns:
        Answer object with conclusion, confidence, and reasoning trace
    """
    # Stage 1: MODE - Classify the problem
    mode = ModeClassifier().classify(prompt)

    # Stage 2: GROUND - Retrieve references from RAG
    retriever = Retriever(embed_dim=config.rag.embed_dim)
    # Seed small corpus for demo (in production, load from config)
    retriever.add_corpus([
        "ANOVA explains variance components and tests mean differences across groups",
        "DOE helps understand factor interactions and optimize processes",
        "SPC monitors process stability using control charts and statistical rules",
        "T-tests compare means between two groups with assumptions of normality",
        "Regression models relationships between variables and predicts outcomes",
        "Western Electric rules detect special cause variation in control charts",
        "Factorial designs explore all combinations of factor levels",
        "P-values indicate statistical significance but should be interpreted with effect size",
    ])
    grounded: GroundedContext = attach_references(prompt, mode, retriever)

    # Stage 3: PLAN - Build execution plan
    plan = Planner().build_plan(prompt, mode)

    # Stage 4: EXECUTE - Run the plan with tools
    executor = Executor()
    exec_result = executor.run_plan(plan, prompt, grounded=grounded)

    # Stage 5: ARGUE - Build arguments from execution results
    arguments = build_arguments(prompt, exec_result=exec_result)

    # Stage 6: ARBITER - Decide with consistency and quality checks
    decision = decide(arguments)

    # Stage 7: AUDIT - Review and adjust confidence
    audit_report = run_audit(decision, arguments)

    # Stage 8: ANSWER - Build final structured answer
    return build_answer(
        prompt=prompt,
        decision=decision,
        audit=audit_report,
        exec_result=exec_result,
        grounded=grounded,
        mode=mode
    )
