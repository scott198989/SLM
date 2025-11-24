from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from havoc_tools.dsl.executor import DSLExecutor, ExecutionOutcome
from havoc_tools.python_math import engine
from havoc_srs.plan import Plan


@dataclass
class ExecutionResult:
    outputs: List[Dict[str, Any]]


class Executor:
    def __init__(self):
        self.dsl_executor = DSLExecutor()

    def run_plan(self, plan: Plan, prompt: str) -> ExecutionResult:
        outputs: List[Dict[str, Any]] = []
        for step in plan.steps:
            if "dsl" in step.tools:
                outcome: ExecutionOutcome = self.dsl_executor.execute(prompt)
                outputs.append({"step": step.description, "outcome": outcome.payload})
            if "python_math" in step.tools:
                # Example math operation; in real use, parse prompt
                ttest = engine.run_ttest([1, 2, 3], [1.1, 1.9, 3.2])
                outputs.append({"step": step.description, "ttest": ttest.__dict__})
        return ExecutionResult(outputs=outputs)
