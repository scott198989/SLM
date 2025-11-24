"""HAVOC-7B SRS (Structured Reasoning Stack) Module

This module implements the structured reasoning pipeline including
argumentation, arbitration, auditing, and execution.
"""

from havoc_srs.orchestrator import run_pipeline
from havoc_srs.answer import Answer, build_answer
from havoc_srs.argue import Argument, build_arguments
from havoc_srs.arbiter import ArbiterDecision, decide
from havoc_srs.audit import AuditReport, run_audit
from havoc_srs.execute import ExecutionResult, Executor
from havoc_srs.ground import GroundedContext, attach_references
from havoc_srs.mode import ModePrediction, TaskType, ModeClassifier
from havoc_srs.plan import Plan, Planner

__all__ = [
    "run_pipeline",
    "Answer",
    "build_answer",
    "Argument",
    "build_arguments",
    "ArbiterDecision",
    "decide",
    "AuditReport",
    "run_audit",
    "ExecutionResult",
    "Executor",
    "GroundedContext",
    "attach_references",
    "ModePrediction",
    "TaskType",
    "ModeClassifier",
    "Plan",
    "Planner",
]
