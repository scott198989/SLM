from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TaskType(Enum):
    MATH = auto()
    STATS = auto()
    DOE = auto()
    SPC = auto()
    PROCESS_ENG = auto()
    MATERIALS = auto()
    GENERAL = auto()


class Difficulty(Enum):
    TRIVIAL = auto()
    NORMAL = auto()
    HARD = auto()
    CRITICAL = auto()


class Risk(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass
class ModePrediction:
    task: TaskType
    difficulty: Difficulty
    risk: Risk


class ModeClassifier:
    def classify(self, prompt: str) -> ModePrediction:
        lowered = prompt.lower()
        if any(tok in lowered for tok in ["anova", "ttest", "regression", "p-value"]):
            task = TaskType.STATS
        elif "doe" in lowered or "factor" in lowered:
            task = TaskType.DOE
        elif "spc" in lowered or "control chart" in lowered:
            task = TaskType.SPC
        else:
            task = TaskType.GENERAL
        difficulty = Difficulty.HARD if len(prompt) > 280 else Difficulty.NORMAL
        risk = Risk.HIGH if task in {TaskType.SPC, TaskType.PROCESS_ENG} else Risk.MEDIUM
        return ModePrediction(task=task, difficulty=difficulty, risk=risk)
