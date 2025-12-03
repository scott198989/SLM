"""
DSL Executor - Domain-Specific Language tool for HAVOC PRIME

Executes DOE/SPC/STATS DSL commands.
"""

from __future__ import annotations

from typing import Any, Dict

# Import existing DSL executor
from havoc_tools.dsl.executor import DSLExecutor as OriginalDSLExecutor
from havoc_tools.dsl.executor import ExecutionOutcome


class DSLExecutor:
    """
    Callable interface to DSL execution.

    Executes structured commands in JSON/YAML format for:
    - DOE operations
    - SPC operations
    - STAT operations
    - MATH operations
    """

    def __init__(self):
        self.executor = OriginalDSLExecutor()

    def execute(self, dsl_command: str) -> Dict[str, Any]:
        """
        Execute a DSL command.

        Args:
            dsl_command: DSL command as JSON/YAML string

        Returns:
            Dict with:
                - success: bool
                - payload: dict with results
                - description: str
                - error: optional str
        """
        try:
            outcome: ExecutionOutcome = self.executor.execute(dsl_command)

            return {
                "success": outcome.success,
                "payload": outcome.payload,
                "description": outcome.description,
                "error": outcome.error
            }

        except Exception as e:
            return {
                "success": False,
                "payload": {},
                "description": "DSL execution failed",
                "error": str(e)
            }

    def parse_dsl_from_prompt(self, prompt: str) -> str:
        """
        Extract DSL from prompt or generate from keywords.

        Args:
            prompt: User's prompt (may contain DSL block)

        Returns:
            Extracted or generated DSL command
        """
        # Look for code blocks
        if "```yaml" in prompt or "```json" in prompt or "```" in prompt:
            lines = prompt.split('\n')
            in_block = False
            dsl_lines = []

            for line in lines:
                if line.strip().startswith("```yaml") or line.strip().startswith("```json") or line.strip() == "```":
                    if in_block:
                        break
                    in_block = True
                    continue
                elif in_block:
                    dsl_lines.append(line)

            if dsl_lines:
                return '\n'.join(dsl_lines)

        # Check if prompt looks like structured data
        prompt_stripped = prompt.strip()
        if prompt_stripped.startswith('{') or ':' in prompt_stripped:
            return prompt_stripped

        # Generate simple DSL from keywords
        return self._generate_dsl_from_keywords(prompt)

    def _generate_dsl_from_keywords(self, prompt: str) -> str:
        """Generate simple DSL from prompt keywords"""
        import json

        lower = prompt.lower()

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
        if 'doe' in lower or 'design of experiment' in lower or 'factorial' in lower:
            return json.dumps({
                "DOE": {
                    "operation": "factorial",
                    "factors": [
                        {"name": "Factor_A", "levels": [-1, 1]},
                        {"name": "Factor_B", "levels": [-1, 1]}
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

        # Math expression
        if any(op in lower for op in ['calculate', 'evaluate', 'compute']):
            return json.dumps({
                "MATH": {
                    "expression": "x**2 + y",
                    "variables": {"x": 2, "y": 3}
                }
            })

        # Default
        return "{}"
