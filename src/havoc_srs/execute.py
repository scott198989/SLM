from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from havoc_tools.dsl.executor import DSLExecutor, ExecutionOutcome
from havoc_tools.python_math import engine
from havoc_srs.ground import GroundedContext
from havoc_srs.plan import Plan


@dataclass
class StepResult:
    step_description: str
    tool: str
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    steps: List[StepResult]
    overall_success: bool
    summary: str


class Executor:
    def __init__(self):
        self.dsl_executor = DSLExecutor()

    def run_plan(self, plan: Plan, prompt: str, grounded: Optional[GroundedContext] = None) -> ExecutionResult:
        """Execute a plan by running each step with appropriate tools."""
        step_results: List[StepResult] = []
        overall_success = True

        for step in plan.steps:
            step_success = True
            step_output = {}
            step_error = None
            tool_used = "none"

            # Execute DSL operations
            if "dsl" in step.tools:
                tool_used = "dsl"
                try:
                    # Try to extract DSL from prompt (look for YAML/JSON blocks)
                    dsl_content = self._extract_dsl_from_prompt(prompt)
                    outcome: ExecutionOutcome = self.dsl_executor.execute(dsl_content)

                    step_output = {
                        "description": outcome.description,
                        "payload": outcome.payload,
                        "success": outcome.success
                    }
                    step_success = outcome.success
                    step_error = outcome.error
                except Exception as e:
                    step_success = False
                    step_error = f"DSL execution failed: {str(e)}"
                    step_output = {"error": step_error}

            # Execute Python math/stats operations
            elif "python_math" in step.tools:
                tool_used = "python_math"
                try:
                    # Parse prompt to extract statistical operation
                    stat_result = self._execute_math_from_prompt(prompt)
                    step_output = stat_result
                    step_success = True
                except Exception as e:
                    step_success = False
                    step_error = f"Math execution failed: {str(e)}"
                    step_output = {"error": step_error}

            # Execute RAG operations
            elif "rag" in step.tools:
                tool_used = "rag"
                try:
                    if grounded:
                        step_output = {
                            "references": grounded.references[:3],  # Top 3
                            "count": len(grounded.references)
                        }
                        step_success = True
                    else:
                        step_output = {"references": [], "count": 0}
                        step_success = True
                except Exception as e:
                    step_success = False
                    step_error = f"RAG execution failed: {str(e)}"
                    step_output = {"error": step_error}

            # Default case
            else:
                tool_used = "none"
                step_output = {"note": "No tool executed for this step"}
                step_success = True

            step_results.append(StepResult(
                step_description=step.description,
                tool=tool_used,
                success=step_success,
                output=step_output,
                error=step_error
            ))

            if not step_success:
                overall_success = False

        # Generate summary
        successful_steps = sum(1 for s in step_results if s.success)
        summary = f"Executed {len(step_results)} steps: {successful_steps} succeeded, {len(step_results) - successful_steps} failed"

        return ExecutionResult(
            steps=step_results,
            overall_success=overall_success,
            summary=summary
        )

    def _extract_dsl_from_prompt(self, prompt: str) -> str:
        """Extract DSL content from prompt or return prompt as-is."""
        # Look for YAML blocks
        if "```yaml" in prompt or "```" in prompt:
            # Extract code block
            lines = prompt.split('\n')
            in_block = False
            dsl_lines = []
            for line in lines:
                if line.strip().startswith("```yaml") or line.strip().startswith("```json"):
                    in_block = True
                    continue
                elif line.strip() == "```":
                    if in_block:
                        break
                elif in_block:
                    dsl_lines.append(line)
            if dsl_lines:
                return '\n'.join(dsl_lines)

        # Check if prompt looks like YAML/JSON
        prompt_stripped = prompt.strip()
        if prompt_stripped.startswith('{') or ':' in prompt_stripped:
            return prompt_stripped

        # Default: create a simple DSL request from keywords
        return self._generate_dsl_from_keywords(prompt)

    def _generate_dsl_from_keywords(self, prompt: str) -> str:
        """Generate simple DSL from prompt keywords."""
        lower = prompt.lower()

        # Math expression
        if any(op in lower for op in ['calculate', 'evaluate', 'compute']):
            # Simple example
            return json.dumps({"MATH": {"expression": "x**2 + y", "variables": {"x": 2, "y": 3}}})

        # T-test
        if 'ttest' in lower or 't-test' in lower:
            return json.dumps({
                "STAT_TEST": {
                    "test_type": "ttest",
                    "data_a": [1, 2, 3, 4, 5],
                    "data_b": [1.5, 2.5, 3.5, 4.5, 5.5]
                }
            })

        # DOE
        if 'doe' in lower or 'design of experiment' in lower:
            return json.dumps({
                "DOE": {
                    "operation": "factorial",
                    "factors": [
                        {"name": "Temperature", "levels": [-1, 1]},
                        {"name": "Pressure", "levels": [-1, 1]}
                    ]
                }
            })

        # SPC
        if 'spc' in lower or 'control chart' in lower:
            return json.dumps({
                "SPC": {
                    "chart_type": "I_MR",
                    "data": list(range(20))
                }
            })

        # Default: return empty operation
        return "{}"

    def _execute_math_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """Execute math/stats operation inferred from prompt."""
        lower = prompt.lower()

        # T-test example
        if 'ttest' in lower or 't-test' in lower:
            result = engine.run_ttest([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5])
            return {"operation": "ttest", "result": asdict(result)}

        # Default: simple calculation
        return {"operation": "none", "note": "No specific math operation identified"}
