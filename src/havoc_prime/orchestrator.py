"""
HAVOC PRIME Orchestrator

Main entry point that coordinates all PRIME components + SRS tools.
"""

from __future__ import annotations

from typing import Any, Dict

from havoc_prime.adversarial import (
    Advocate,
    AdversarialSynthesizer,
    HavocAttack,
    Pragmatist,
)
from havoc_prime.chrono_loop import ChronoLoop
from havoc_prime.compression import FinalCompression
from havoc_prime.constraints import ConstraintBackbone
from havoc_prime.operator_graph import OperatorGraph, OperatorGraphBuilder
from havoc_prime.router import Budget, RoutingDecision, TaskRouter
from havoc_prime.verification import GlobalVerification
from havoc_prime.workspace import GlobalWorkspace
from srs_tools import AnswerFormatter, DSLExecutor, PythonMathEngine, RAGHelper


class HavocPrimeOrchestrator:
    """
    Main orchestrator for HAVOC HYBRID system.

    Flow:
    1. Route task → classify + assign budget
    2. If MICRO → direct answer (no PRIME)
    3. Build operator graph (subgoals)
    4. Initialize workspace + constraints
    5. For each subgoal:
       - Apply chrono-loop (if MEDIUM/HEAVY)
       - Run adversarial reasoning (Advocate/Attack/Pragmatist)
       - Call SRS tools if needed
       - Synthesize local result
       - Update workspace
    6. Global verification
    7. Final compression
    8. Format answer

    Args:
        enable_chrono: Enable chrono-loop (default: True)
        enable_adversarial: Enable triple-fork reasoning (default: True)
        max_chrono_iterations: Max chrono-loop iterations (default: 3)
    """

    def __init__(
        self,
        enable_chrono: bool = True,
        enable_adversarial: bool = True,
        max_chrono_iterations: int = 3
    ):
        # Initialize components
        self.router = TaskRouter()
        self.graph_builder = OperatorGraphBuilder()
        self.chrono_loop = ChronoLoop(max_iterations=max_chrono_iterations)
        self.advocate = Advocate()
        self.havoc_attack = HavocAttack()
        self.pragmatist = Pragmatist()
        self.synthesizer = AdversarialSynthesizer()
        self.verification = GlobalVerification()
        self.compression = FinalCompression()

        # SRS tools
        self.dsl_executor = DSLExecutor()
        self.python_math = PythonMathEngine()
        self.rag_helper = RAGHelper()
        self.answer_formatter = AnswerFormatter()

        # Flags
        self.enable_chrono = enable_chrono
        self.enable_adversarial = enable_adversarial

    def process(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point - process user prompt.

        Args:
            prompt: User's question or task

        Returns:
            Dict with formatted answer
        """
        # Step 1: Route task
        routing: RoutingDecision = self.router.route(prompt)

        # Step 2: Handle MICRO budget (direct answer, no reasoning)
        if routing.budget == Budget.MICRO:
            return self._handle_micro_budget(prompt)

        # Step 3: Build operator graph
        operator_graph: OperatorGraph = self.graph_builder.build_graph(
            prompt, routing.task_type, routing.budget
        )

        # Step 4: Initialize workspace and constraints
        workspace = GlobalWorkspace()
        constraint_backbone = ConstraintBackbone(routing.task_type)

        # Add global constraints to workspace
        for constraint_desc in operator_graph.global_constraints:
            workspace.add_constraint(constraint_desc)

        # Step 5: Execute subgoals
        for subgoal in operator_graph.subgoals:
            subgoal.status = "in_progress"

            # Check dependencies
            ready_subgoals = operator_graph.get_ready_subgoals()
            if subgoal not in ready_subgoals:
                continue

            # Execute subgoal
            subgoal_result = self._execute_subgoal(
                subgoal, workspace, routing.budget, prompt
            )

            # Store result
            subgoal.result = subgoal_result
            subgoal.confidence = subgoal_result.get("confidence", 0.5)
            subgoal.status = "completed" if subgoal_result.get("success", True) else "failed"

            # Update workspace
            workspace.store_partial_result(subgoal.id, subgoal_result)

        # Step 6: Enforce constraints
        constraint_violations = constraint_backbone.enforce(workspace)

        # Step 7: Global verification
        verification_report = self.verification.verify(workspace, operator_graph)

        # Apply confidence penalty from verification
        workspace.update_global_confidence(
            workspace.global_confidence * (1.0 - verification_report.confidence_penalty)
        )

        # Step 8: Final synthesis (global adversarial reasoning if HEAVY)
        if routing.budget == Budget.HEAVY and self.enable_adversarial:
            final_result = self._global_adversarial_synthesis(workspace, operator_graph)
        else:
            final_result = {"conclusion": "Analysis completed", "confidence": workspace.global_confidence}

        # Step 9: Final compression
        compressed_result = self.compression.compress(workspace, operator_graph, final_result)

        # Step 10: Format answer
        formatted_answer = self.answer_formatter.format_answer(
            prompt=prompt,
            workspace=workspace,
            operator_graph=operator_graph,
            final_result=compressed_result,
            task_type=routing.task_type.name
        )

        return {
            "answer": formatted_answer.format_human_readable(),
            "routing": {
                "task_type": routing.task_type.name,
                "budget": routing.budget.name,
                "difficulty": routing.difficulty.name,
                "risk": routing.risk.name
            },
            "workspace_summary": workspace.summarize(),
            "verification": {
                "passed": verification_report.passed,
                "issues": verification_report.issues,
                "warnings": verification_report.warnings
            },
            "formatted_answer_object": formatted_answer  # For programmatic access
        }

    def _handle_micro_budget(self, prompt: str) -> Dict[str, Any]:
        """Handle MICRO budget - direct answer without reasoning"""
        # Simple direct evaluation
        prompt_lower = prompt.lower().strip()

        # Try simple arithmetic
        if any(op in prompt_lower for op in ['+', '-', '*', '/']):
            try:
                # Extract and evaluate (VERY UNSAFE - just for demo)
                # In production, use a proper math parser
                result = eval(prompt_lower)
                conclusion = f"Result: {result}"
                confidence = 1.0
            except:
                conclusion = f"Cannot parse: {prompt}"
                confidence = 0.0
        else:
            # Definitional query
            conclusion = f"Query: {prompt}\nAnswer: [Direct lookup would go here]"
            confidence = 0.5

        return {
            "answer": f"## CONCLUSION\n{conclusion}\n\n## CONFIDENCE\n{confidence * 100:.1f}%",
            "routing": {
                "task_type": "TRIVIAL",
                "budget": "MICRO",
                "difficulty": "TRIVIAL",
                "risk": "LOW"
            },
            "workspace_summary": {"facts_count": 0},
            "verification": {"passed": True, "issues": [], "warnings": []}
        }

    def _execute_subgoal(
        self,
        subgoal,
        workspace: GlobalWorkspace,
        budget: Budget,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Execute a single subgoal.

        This is where SRS tools are called.
        """
        result = {"success": True, "confidence": 0.5}

        # Step A: Apply chrono-loop if enabled and budget allows
        if self.enable_chrono and budget in {Budget.MEDIUM, Budget.HEAVY}:
            result = self.chrono_loop.run(result, noise_level=0.1)

        # Step B: Call SRS tools if needed
        if "dsl_executor" in subgoal.required_tools:
            dsl_result = self._call_dsl_tool(prompt)
            result.update(dsl_result)

        if "python_math" in subgoal.required_tools:
            math_result = self._call_math_tool(subgoal.description)
            result.update(math_result)

        if "rag_helper" in subgoal.required_tools:
            rag_result = self._call_rag_tool(prompt)
            result.update(rag_result)

        # Step C: Adversarial reasoning (if enabled and budget allows)
        if self.enable_adversarial and budget in {Budget.MEDIUM, Budget.HEAVY}:
            advocate_arg = self.advocate.build_argument(result, workspace)
            attack_arg = self.havoc_attack.build_argument(result, workspace)
            pragmatist_arg = self.pragmatist.build_argument(result, workspace)

            synthesis = self.synthesizer.synthesize(advocate_arg, attack_arg, pragmatist_arg)

            result["adversarial_synthesis"] = {
                "conclusion": synthesis.conclusion,
                "confidence": synthesis.confidence,
                "rationale": synthesis.rationale
            }
            result["confidence"] = synthesis.confidence

        return result

    def _call_dsl_tool(self, prompt: str) -> Dict[str, Any]:
        """Call DSL executor tool"""
        try:
            dsl_command = self.dsl_executor.parse_dsl_from_prompt(prompt)
            result = self.dsl_executor.execute(dsl_command)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _call_math_tool(self, subgoal_description: str) -> Dict[str, Any]:
        """Call Python math tool"""
        try:
            # Infer operation from description
            desc_lower = subgoal_description.lower()

            if "ttest" in desc_lower or "t-test" in desc_lower:
                # Dummy t-test
                result = self.python_math.t_test([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5])
                return result

            elif "anova" in desc_lower:
                # Dummy ANOVA
                result = self.python_math.anova([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
                return result

            else:
                return {"success": True, "note": "Math tool placeholder"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _call_rag_tool(self, prompt: str) -> Dict[str, Any]:
        """Call RAG helper tool"""
        try:
            references = self.rag_helper.retrieve(prompt, k=3)
            return {
                "success": True,
                "references": references,
                "reference_count": len(references)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _global_adversarial_synthesis(
        self,
        workspace: GlobalWorkspace,
        operator_graph: OperatorGraph
    ) -> Dict[str, Any]:
        """Run global adversarial reasoning on full result"""
        # Aggregate all subgoal results
        aggregated_result = {}

        for subgoal in operator_graph.subgoals:
            if subgoal.result:
                aggregated_result.update(subgoal.result)

        # Run adversarial reasoning
        advocate_arg = self.advocate.build_argument(aggregated_result, workspace)
        attack_arg = self.havoc_attack.build_argument(aggregated_result, workspace)
        pragmatist_arg = self.pragmatist.build_argument(aggregated_result, workspace)

        synthesis = self.synthesizer.synthesize(advocate_arg, attack_arg, pragmatist_arg)

        return {
            "conclusion": synthesis.conclusion,
            "confidence": synthesis.confidence,
            "rationale": synthesis.rationale,
            "key_numbers": aggregated_result
        }


# Convenience function
def run_havoc_prime(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run HAVOC PRIME.

    Args:
        prompt: User's question
        **kwargs: Additional options (enable_chrono, enable_adversarial, etc.)

    Returns:
        Result dict with answer
    """
    orchestrator = HavocPrimeOrchestrator(**kwargs)
    return orchestrator.process(prompt)
