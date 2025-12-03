"""
Task Router & Budget Allocator for HAVOC PRIME

Classifies incoming tasks and assigns computational budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TaskType(Enum):
    """Domain-specific task types"""
    MATH = auto()
    STATS = auto()
    DOE = auto()
    SPC = auto()
    PROCESS_ENG = auto()
    MATERIALS = auto()
    GENERAL = auto()
    TRIVIAL = auto()


class Budget(Enum):
    """Computational budget levels"""
    MICRO = auto()      # Direct answer, no reasoning
    LIGHT = auto()      # Minimal PRIME (no loops, no adversaries)
    MEDIUM = auto()     # Standard PRIME (limited loops)
    HEAVY = auto()      # Full PRIME + SRS tools


class Difficulty(Enum):
    """Task difficulty assessment"""
    TRIVIAL = auto()
    SIMPLE = auto()
    MODERATE = auto()
    HARD = auto()
    CRITICAL = auto()


class Risk(Enum):
    """Risk level for incorrect answers"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class RoutingDecision:
    """Router output"""
    task_type: TaskType
    budget: Budget
    difficulty: Difficulty
    risk: Risk
    reasoning: str


class TaskRouter:
    """
    Routes tasks and assigns budgets.

    Budget assignment logic:
    - MICRO: Trivial arithmetic, lookups (3+3, "what is ANOVA?")
    - LIGHT: Simple questions, single-step reasoning
    - MEDIUM: Multi-step problems, moderate complexity
    - HEAVY: Complex domain tasks (DOE design, SPC analysis, multi-factor problems)
    """

    def __init__(self):
        self.trivial_patterns = [
            "what is", "define", "explain", "who", "when", "where"
        ]
        self.heavy_patterns = [
            "design", "optimize", "analyze", "compare multiple", "full analysis"
        ]

    def route(self, prompt: str) -> RoutingDecision:
        """
        Route task and assign budget.

        Args:
            prompt: User's question or task

        Returns:
            RoutingDecision with task type, budget, difficulty, and risk
        """
        prompt_lower = prompt.lower()
        prompt_len = len(prompt)

        # Step 1: Classify task type
        task_type = self._classify_task_type(prompt_lower)

        # Step 2: Assess difficulty
        difficulty = self._assess_difficulty(prompt_lower, prompt_len)

        # Step 3: Assess risk
        risk = self._assess_risk(task_type, difficulty)

        # Step 4: Assign budget
        budget = self._assign_budget(task_type, difficulty, risk, prompt_lower)

        # Step 5: Build reasoning
        reasoning = self._build_reasoning(task_type, budget, difficulty, risk)

        return RoutingDecision(
            task_type=task_type,
            budget=budget,
            difficulty=difficulty,
            risk=risk,
            reasoning=reasoning
        )

    def _classify_task_type(self, prompt_lower: str) -> TaskType:
        """Classify domain task type"""
        # Check for trivial patterns first
        if any(pat in prompt_lower for pat in ["what is", "define ", "who ", "when ", "where "]):
            return TaskType.TRIVIAL

        # Domain-specific classification
        if any(kw in prompt_lower for kw in ["anova", "ttest", "t-test", "regression", "p-value", "hypothesis"]):
            return TaskType.STATS

        if any(kw in prompt_lower for kw in ["doe", "design of experiment", "factorial", "box-behnken", "central composite"]):
            return TaskType.DOE

        if any(kw in prompt_lower for kw in ["spc", "control chart", "control limit", "western electric", "process control"]):
            return TaskType.SPC

        if any(kw in prompt_lower for kw in ["manufacturing", "process", "yield", "throughput", "capability"]):
            return TaskType.PROCESS_ENG

        if any(kw in prompt_lower for kw in ["material", "alloy", "metallurgy", "strength", "hardness"]):
            return TaskType.MATERIALS

        if any(kw in prompt_lower for kw in ["calculate", "compute", "evaluate", "+", "-", "*", "/", "^"]):
            return TaskType.MATH

        return TaskType.GENERAL

    def _assess_difficulty(self, prompt_lower: str, prompt_len: int) -> Difficulty:
        """Assess task difficulty"""
        # Length heuristic
        if prompt_len < 30:
            base_difficulty = Difficulty.TRIVIAL
        elif prompt_len < 100:
            base_difficulty = Difficulty.SIMPLE
        elif prompt_len < 250:
            base_difficulty = Difficulty.MODERATE
        else:
            base_difficulty = Difficulty.HARD

        # Pattern-based adjustments
        if any(pat in prompt_lower for pat in ["design", "optimize", "full analysis", "comprehensive"]):
            if base_difficulty.value < Difficulty.HARD.value:
                base_difficulty = Difficulty.HARD

        if any(pat in prompt_lower for pat in ["critical", "production", "safety", "failure"]):
            base_difficulty = Difficulty.CRITICAL

        # Complexity indicators
        if "and" in prompt_lower and "or" in prompt_lower:
            # Multiple conditions increase difficulty
            if base_difficulty.value < Difficulty.MODERATE.value:
                base_difficulty = Difficulty.MODERATE

        return base_difficulty

    def _assess_risk(self, task_type: TaskType, difficulty: Difficulty) -> Risk:
        """Assess risk of incorrect answer"""
        # High-risk domains
        high_risk_tasks = {TaskType.SPC, TaskType.PROCESS_ENG, TaskType.DOE}

        if task_type in high_risk_tasks:
            if difficulty in {Difficulty.CRITICAL, Difficulty.HARD}:
                return Risk.CRITICAL
            return Risk.HIGH

        if task_type == TaskType.STATS:
            if difficulty == Difficulty.CRITICAL:
                return Risk.HIGH
            return Risk.MEDIUM

        if difficulty == Difficulty.CRITICAL:
            return Risk.HIGH

        if difficulty in {Difficulty.TRIVIAL, Difficulty.SIMPLE}:
            return Risk.LOW

        return Risk.MEDIUM

    def _assign_budget(self, task_type: TaskType, difficulty: Difficulty, risk: Risk, prompt_lower: str) -> Budget:
        """Assign computational budget"""
        # MICRO: Trivial tasks only
        if task_type == TaskType.TRIVIAL and difficulty == Difficulty.TRIVIAL:
            return Budget.MICRO

        if difficulty == Difficulty.TRIVIAL and risk == Risk.LOW:
            # Simple arithmetic like "3+3"
            if any(op in prompt_lower for op in [" + ", " - ", " * ", " / "]):
                if len(prompt_lower) < 20:
                    return Budget.MICRO

        # HEAVY: Complex domain tasks
        if task_type in {TaskType.DOE, TaskType.SPC} and difficulty in {Difficulty.HARD, Difficulty.CRITICAL}:
            return Budget.HEAVY

        if risk in {Risk.HIGH, Risk.CRITICAL}:
            return Budget.HEAVY

        if any(kw in prompt_lower for kw in ["design", "optimize", "full analysis", "comprehensive analysis"]):
            return Budget.HEAVY

        # MEDIUM: Standard reasoning tasks
        if difficulty in {Difficulty.MODERATE, Difficulty.HARD}:
            return Budget.MEDIUM

        if task_type in {TaskType.STATS, TaskType.PROCESS_ENG}:
            return Budget.MEDIUM

        # LIGHT: Everything else
        return Budget.LIGHT

    def _build_reasoning(self, task_type: TaskType, budget: Budget, difficulty: Difficulty, risk: Risk) -> str:
        """Build human-readable reasoning for routing decision"""
        parts = [
            f"Task classified as {task_type.name}",
            f"Difficulty: {difficulty.name}",
            f"Risk: {risk.name}",
            f"Budget: {budget.name}"
        ]

        if budget == Budget.MICRO:
            parts.append("Direct answer - no reasoning required")
        elif budget == Budget.LIGHT:
            parts.append("Light reasoning - minimal subgoals")
        elif budget == Budget.MEDIUM:
            parts.append("Standard reasoning - moderate depth")
        elif budget == Budget.HEAVY:
            parts.append("Full reasoning - deep analysis with adversarial checks")

        return " | ".join(parts)
