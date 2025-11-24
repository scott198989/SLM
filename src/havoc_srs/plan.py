from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from havoc_srs.mode import ModePrediction, TaskType


@dataclass
class PlanStep:
    description: str
    tools: List[str] = field(default_factory=list)


@dataclass
class Plan:
    problem_type: TaskType
    steps: List[PlanStep]
    required_outputs: List[str]
    risk_level: str


class Planner:
    def build_plan(self, prompt: str, mode: ModePrediction) -> Plan:
        if mode.task == TaskType.DOE:
            steps = [
                PlanStep("Parse DOE DSL", tools=["dsl"]),
                PlanStep("Run DOE analysis", tools=["python_math"]),
            ]
        elif mode.task == TaskType.STATS:
            steps = [PlanStep("Identify statistical test", tools=["python_math"])]
        else:
            steps = [PlanStep("Retrieve references", tools=["rag"]), PlanStep("Draft response")]
        required_outputs = ["summary", "confidence"]
        return Plan(problem_type=mode.task, steps=steps, required_outputs=required_outputs, risk_level=mode.risk.name)
