"""
Operator Graph & Subgoal Planner for HAVOC PRIME

Decomposes complex tasks into dependency-aware subgoals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from havoc_prime.router import Budget, TaskType


@dataclass
class Subgoal:
    """
    A single subgoal in the operator graph.

    Attributes:
        id: Unique identifier
        description: Human-readable description
        dependencies: IDs of subgoals that must complete first
        required_tools: SRS tools needed (e.g., ["dsl_executor", "python_math"])
        constraints: Domain constraints
        expected_output: Description of expected result
        status: pending, in_progress, completed, failed
        result: Output from execution (populated after completion)
        confidence: Confidence in result (0-1)
    """
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    expected_output: str = ""
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0


@dataclass
class OperatorGraph:
    """
    Directed acyclic graph of subgoals.

    Attributes:
        goal: Main goal description
        subgoals: List of subgoals
        global_constraints: Constraints that apply to all subgoals
        max_subgoals: Maximum allowed subgoals (budget-dependent)
    """
    goal: str
    subgoals: List[Subgoal] = field(default_factory=list)
    global_constraints: List[str] = field(default_factory=list)
    max_subgoals: int = 7

    def add_subgoal(self, subgoal: Subgoal) -> None:
        """Add subgoal to graph"""
        if len(self.subgoals) >= self.max_subgoals:
            raise ValueError(f"Cannot exceed {self.max_subgoals} subgoals")
        self.subgoals.append(subgoal)

    def get_ready_subgoals(self) -> List[Subgoal]:
        """Get subgoals with satisfied dependencies"""
        completed_ids = {sg.id for sg in self.subgoals if sg.status == "completed"}

        ready = []
        for sg in self.subgoals:
            if sg.status == "pending":
                if all(dep in completed_ids for dep in sg.dependencies):
                    ready.append(sg)

        return ready

    def get_subgoal(self, subgoal_id: str) -> Optional[Subgoal]:
        """Get subgoal by ID"""
        for sg in self.subgoals:
            if sg.id == subgoal_id:
                return sg
        return None

    def is_complete(self) -> bool:
        """Check if all subgoals are completed"""
        return all(sg.status in {"completed", "failed"} for sg in self.subgoals)

    def validate_dag(self) -> bool:
        """Validate that graph is a DAG (no cycles)"""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            node = self.get_subgoal(node_id)
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for sg in self.subgoals:
            if sg.id not in visited:
                if has_cycle(sg.id):
                    return False

        return True


class OperatorGraphBuilder:
    """
    Builds operator graphs based on task type and budget.

    Graph complexity is budget-dependent:
    - LIGHT: max 3 subgoals
    - MEDIUM: max 7 subgoals
    - HEAVY: max 12 subgoals
    """

    def __init__(self):
        self.max_subgoals_by_budget = {
            Budget.MICRO: 0,    # No graph
            Budget.LIGHT: 3,
            Budget.MEDIUM: 7,
            Budget.HEAVY: 12
        }

    def build_graph(self, prompt: str, task_type: TaskType, budget: Budget) -> OperatorGraph:
        """
        Build operator graph for task.

        Args:
            prompt: User's question
            task_type: Classified task type
            budget: Computational budget

        Returns:
            OperatorGraph with subgoals
        """
        max_subgoals = self.max_subgoals_by_budget.get(budget, 7)

        graph = OperatorGraph(
            goal=prompt,
            max_subgoals=max_subgoals,
            global_constraints=self._get_global_constraints(task_type)
        )

        # Build subgoals based on task type
        if budget == Budget.MICRO:
            # No subgoals for MICRO
            return graph

        if task_type == TaskType.DOE:
            self._build_doe_graph(graph, prompt)
        elif task_type == TaskType.SPC:
            self._build_spc_graph(graph, prompt)
        elif task_type == TaskType.STATS:
            self._build_stats_graph(graph, prompt)
        elif task_type == TaskType.PROCESS_ENG:
            self._build_process_graph(graph, prompt)
        else:
            self._build_general_graph(graph, prompt, budget)

        # Validate DAG
        if not graph.validate_dag():
            raise ValueError("Operator graph contains cycles")

        return graph

    def _get_global_constraints(self, task_type: TaskType) -> List[str]:
        """Get domain-specific global constraints"""
        constraints = []

        if task_type == TaskType.STATS:
            constraints.extend([
                "Check statistical assumptions",
                "Report effect size, not just p-value",
                "Consider sample size adequacy"
            ])
        elif task_type == TaskType.DOE:
            constraints.extend([
                "Verify design balance",
                "Check for confounding",
                "Validate factor independence"
            ])
        elif task_type == TaskType.SPC:
            constraints.extend([
                "Apply control chart rules consistently",
                "Validate subgroup rationale",
                "Check for special cause variation"
            ])

        return constraints

    def _build_doe_graph(self, graph: OperatorGraph, prompt: str) -> None:
        """Build DOE-specific subgoals"""
        graph.add_subgoal(Subgoal(
            id="doe_1",
            description="Identify factors and levels from prompt",
            required_tools=[],
            expected_output="List of factors with levels"
        ))

        graph.add_subgoal(Subgoal(
            id="doe_2",
            description="Select appropriate DOE design type",
            dependencies=["doe_1"],
            required_tools=[],
            expected_output="Design type (factorial, Box-Behnken, etc.)"
        ))

        graph.add_subgoal(Subgoal(
            id="doe_3",
            description="Generate design matrix",
            dependencies=["doe_2"],
            required_tools=["dsl_executor"],
            expected_output="Design matrix with run order"
        ))

        if len(graph.subgoals) < graph.max_subgoals:
            graph.add_subgoal(Subgoal(
                id="doe_4",
                description="Validate design properties",
                dependencies=["doe_3"],
                required_tools=["python_math"],
                expected_output="Balance check, confounding analysis"
            ))

    def _build_spc_graph(self, graph: OperatorGraph, prompt: str) -> None:
        """Build SPC-specific subgoals"""
        graph.add_subgoal(Subgoal(
            id="spc_1",
            description="Identify chart type and data structure",
            required_tools=[],
            expected_output="Chart type (I-MR, Xbar-R, etc.)"
        ))

        graph.add_subgoal(Subgoal(
            id="spc_2",
            description="Calculate control limits",
            dependencies=["spc_1"],
            required_tools=["python_math"],
            expected_output="UCL, CL, LCL values"
        ))

        graph.add_subgoal(Subgoal(
            id="spc_3",
            description="Apply Western Electric rules",
            dependencies=["spc_2"],
            required_tools=["python_math"],
            expected_output="Control violations list"
        ))

    def _build_stats_graph(self, graph: OperatorGraph, prompt: str) -> None:
        """Build stats-specific subgoals"""
        graph.add_subgoal(Subgoal(
            id="stats_1",
            description="Identify statistical test needed",
            required_tools=[],
            expected_output="Test type (t-test, ANOVA, regression, etc.)"
        ))

        graph.add_subgoal(Subgoal(
            id="stats_2",
            description="Execute statistical test",
            dependencies=["stats_1"],
            required_tools=["python_math"],
            expected_output="Test statistic, p-value, effect size"
        ))

        if len(graph.subgoals) < graph.max_subgoals:
            graph.add_subgoal(Subgoal(
                id="stats_3",
                description="Interpret results with assumptions",
                dependencies=["stats_2"],
                required_tools=[],
                expected_output="Conclusion with caveats"
            ))

    def _build_process_graph(self, graph: OperatorGraph, prompt: str) -> None:
        """Build process engineering subgoals"""
        graph.add_subgoal(Subgoal(
            id="proc_1",
            description="Identify process metrics and constraints",
            required_tools=["rag_helper"],
            expected_output="List of relevant metrics"
        ))

        graph.add_subgoal(Subgoal(
            id="proc_2",
            description="Analyze process capability",
            dependencies=["proc_1"],
            required_tools=["python_math"],
            expected_output="Cp, Cpk values"
        ))

    def _build_general_graph(self, graph: OperatorGraph, prompt: str, budget: Budget) -> None:
        """Build general-purpose subgoals"""
        # Minimal graph for LIGHT
        if budget == Budget.LIGHT:
            graph.add_subgoal(Subgoal(
                id="gen_1",
                description="Retrieve relevant context",
                required_tools=["rag_helper"],
                expected_output="Reference materials"
            ))

            graph.add_subgoal(Subgoal(
                id="gen_2",
                description="Formulate answer",
                dependencies=["gen_1"],
                required_tools=[],
                expected_output="Draft answer"
            ))

        # Medium complexity
        elif budget == Budget.MEDIUM:
            graph.add_subgoal(Subgoal(
                id="gen_1",
                description="Decompose problem",
                required_tools=[],
                expected_output="Problem components"
            ))

            graph.add_subgoal(Subgoal(
                id="gen_2",
                description="Gather context and data",
                dependencies=["gen_1"],
                required_tools=["rag_helper"],
                expected_output="Relevant information"
            ))

            graph.add_subgoal(Subgoal(
                id="gen_3",
                description="Synthesize solution",
                dependencies=["gen_2"],
                required_tools=[],
                expected_output="Comprehensive answer"
            ))

        # Heavy analysis
        else:
            graph.add_subgoal(Subgoal(
                id="gen_1",
                description="Analyze problem structure",
                required_tools=["rag_helper"],
                expected_output="Problem taxonomy"
            ))

            graph.add_subgoal(Subgoal(
                id="gen_2",
                description="Identify constraints and requirements",
                dependencies=["gen_1"],
                required_tools=[],
                expected_output="Constraint list"
            ))

            graph.add_subgoal(Subgoal(
                id="gen_3",
                description="Generate candidate solutions",
                dependencies=["gen_2"],
                required_tools=[],
                expected_output="Solution alternatives"
            ))

            graph.add_subgoal(Subgoal(
                id="gen_4",
                description="Evaluate and select solution",
                dependencies=["gen_3"],
                required_tools=[],
                expected_output="Selected solution with rationale"
            ))
