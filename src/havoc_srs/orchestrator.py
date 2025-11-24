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
    mode = ModeClassifier().classify(prompt)
    retriever = Retriever(embed_dim=config.rag.embed_dim)
    # seed small corpus for demo
    retriever.add_corpus([
        "ANOVA explains variance components",
        "DOE helps understand factor interactions",
        "SPC monitors process stability",
    ])
    grounded: GroundedContext = attach_references(prompt, mode, retriever)
    plan = Planner().build_plan(prompt, mode)
    exec_result = Executor().run_plan(plan, prompt)
    arguments = build_arguments(prompt)
    decision = decide(arguments)
    audit_report = run_audit(decision, arguments)
    return build_answer(prompt, decision, audit_report, exec_result, grounded)
