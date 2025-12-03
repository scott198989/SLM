"""
Tool Interface for HAVOC-7B

Provides a registry and execution interface for external tools (Python math, DSL, RAG, etc.)
Tools are called via JSON format: {"tool": "<name>", "args": {...}}
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


@dataclass
class ToolResult:
    """Result from tool execution"""
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def success(cls, result: Any, metadata: Dict[str, Any] = None) -> "ToolResult":
        """Create success result"""
        return cls(status=ToolStatus.SUCCESS, result=result, metadata=metadata)

    @classmethod
    def failure(cls, error: str, metadata: Dict[str, Any] = None) -> "ToolResult":
        """Create failure result"""
        return cls(status=ToolStatus.FAILURE, error=error, metadata=metadata)


class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult object
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool arguments.

        Returns:
            JSON schema dict
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class PythonMathTool(BaseTool):
    """
    Tool for Python-based mathematical operations (scipy, statsmodels)

    Supported operations:
    - t_test: Independent two-sample t-test
    - anova: One-way ANOVA
    - regression: Linear regression
    - correlation: Pearson/Spearman correlation
    """

    def __init__(self):
        super().__init__(
            name="python_math",
            description="Execute statistical operations using scipy/statsmodels"
        )

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a math operation"""
        try:
            if operation == "t_test":
                return self._t_test(kwargs.get("group1"), kwargs.get("group2"))
            elif operation == "anova":
                return self._anova(kwargs.get("groups"))
            elif operation == "regression":
                return self._regression(kwargs.get("x"), kwargs.get("y"))
            elif operation == "correlation":
                return self._correlation(
                    kwargs.get("x"),
                    kwargs.get("y"),
                    kwargs.get("method", "pearson")
                )
            else:
                return ToolResult.failure(f"Unknown operation: {operation}")

        except Exception as e:
            return ToolResult.failure(f"Execution error: {str(e)}")

    def _t_test(self, group1: List[float], group2: List[float]) -> ToolResult:
        """Perform independent two-sample t-test"""
        try:
            from scipy import stats

            statistic, pvalue = stats.ttest_ind(group1, group2)
            significant = pvalue < 0.05

            result = {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant": significant,
                "alpha": 0.05,
                "interpretation": "Reject H0" if significant else "Fail to reject H0"
            }

            return ToolResult.success(result, metadata={
                "operation": "t_test",
                "n1": len(group1),
                "n2": len(group2)
            })

        except Exception as e:
            return ToolResult.failure(f"t-test failed: {str(e)}")

    def _anova(self, groups: List[List[float]]) -> ToolResult:
        """Perform one-way ANOVA"""
        try:
            from scipy import stats

            f_statistic, pvalue = stats.f_oneway(*groups)
            significant = pvalue < 0.05

            result = {
                "f_statistic": float(f_statistic),
                "pvalue": float(pvalue),
                "significant": significant,
                "alpha": 0.05,
                "interpretation": "Groups differ significantly" if significant else "No significant difference"
            }

            return ToolResult.success(result, metadata={
                "operation": "anova",
                "num_groups": len(groups),
                "total_n": sum(len(g) for g in groups)
            })

        except Exception as e:
            return ToolResult.failure(f"ANOVA failed: {str(e)}")

    def _regression(self, x: List[float], y: List[float]) -> ToolResult:
        """Perform linear regression"""
        try:
            from scipy import stats
            import numpy as np

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            result = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "equation": f"y = {slope:.4f}x + {intercept:.4f}"
            }

            return ToolResult.success(result, metadata={
                "operation": "regression",
                "n": len(x)
            })

        except Exception as e:
            return ToolResult.failure(f"Regression failed: {str(e)}")

    def _correlation(self, x: List[float], y: List[float], method: str = "pearson") -> ToolResult:
        """Calculate correlation"""
        try:
            from scipy import stats

            if method == "pearson":
                corr, pvalue = stats.pearsonr(x, y)
            elif method == "spearman":
                corr, pvalue = stats.spearmanr(x, y)
            else:
                return ToolResult.failure(f"Unknown method: {method}")

            result = {
                "correlation": float(corr),
                "pvalue": float(pvalue),
                "method": method,
                "significant": pvalue < 0.05
            }

            return ToolResult.success(result, metadata={
                "operation": "correlation",
                "n": len(x)
            })

        except Exception as e:
            return ToolResult.failure(f"Correlation failed: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["t_test", "anova", "regression", "correlation"]
                },
                "group1": {"type": "array", "items": {"type": "number"}},
                "group2": {"type": "array", "items": {"type": "number"}},
                "groups": {"type": "array", "items": {"type": "array"}},
                "x": {"type": "array", "items": {"type": "number"}},
                "y": {"type": "array", "items": {"type": "number"}},
                "method": {"type": "string", "enum": ["pearson", "spearman"]}
            },
            "required": ["operation"]
        }


class DSLExecutorTool(BaseTool):
    """
    Tool for DSL-based DOE/SPC operations

    Supported operations:
    - factorial: Full factorial design
    - box_behnken: Box-Behnken design
    - central_composite: Central Composite Design
    - control_chart: Generate control chart
    """

    def __init__(self):
        super().__init__(
            name="dsl_executor",
            description="Execute DOE/SPC operations via DSL"
        )

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute DSL operation"""
        try:
            # Import DSL executor from srs_tools (if available)
            # For now, provide mock implementation
            if operation == "box_behnken":
                return self._box_behnken(kwargs.get("factors", 3))
            elif operation == "factorial":
                return self._factorial(kwargs.get("factors", []))
            elif operation == "central_composite":
                return self._central_composite(kwargs.get("factors", 3))
            else:
                return ToolResult.failure(f"Unknown DSL operation: {operation}")

        except Exception as e:
            return ToolResult.failure(f"DSL execution error: {str(e)}")

    def _box_behnken(self, factors: int) -> ToolResult:
        """Generate Box-Behnken design"""
        # Mock implementation - replace with actual DSL executor
        runs = 12 + (factors - 2) * 3

        result = {
            "design_type": "box_behnken",
            "factors": factors,
            "runs": runs,
            "levels": [-1, 0, 1],
            "note": "Box-Behnken design does not include corner points"
        }

        return ToolResult.success(result, metadata={
            "operation": "box_behnken"
        })

    def _factorial(self, factor_list: List[Dict[str, Any]]) -> ToolResult:
        """Generate full factorial design"""
        num_factors = len(factor_list)
        runs = 2 ** num_factors

        result = {
            "design_type": "full_factorial",
            "factors": num_factors,
            "runs": runs,
            "factor_names": [f["name"] for f in factor_list]
        }

        return ToolResult.success(result, metadata={
            "operation": "factorial"
        })

    def _central_composite(self, factors: int) -> ToolResult:
        """Generate Central Composite Design"""
        # CCD runs = 2^k + 2k + center points
        factorial_points = 2 ** factors
        axial_points = 2 * factors
        center_points = 5  # Typical
        runs = factorial_points + axial_points + center_points

        result = {
            "design_type": "central_composite",
            "factors": factors,
            "runs": runs,
            "components": {
                "factorial": factorial_points,
                "axial": axial_points,
                "center": center_points
            }
        }

        return ToolResult.success(result, metadata={
            "operation": "central_composite"
        })

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema"""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["factorial", "box_behnken", "central_composite", "control_chart"]
                },
                "factors": {"type": "integer"},
                "factor_list": {"type": "array"}
            },
            "required": ["operation"]
        }


class RAGHelperTool(BaseTool):
    """
    Tool for RAG-based retrieval

    Retrieves relevant documents/snippets from knowledge base
    """

    def __init__(self):
        super().__init__(
            name="rag_helper",
            description="Retrieve relevant information from knowledge base"
        )

    def execute(self, query: str, top_k: int = 3, **kwargs) -> ToolResult:
        """Execute RAG retrieval"""
        try:
            # Mock implementation - replace with actual RAG system
            references = self._mock_retrieve(query, top_k)

            result = {
                "query": query,
                "references": references,
                "count": len(references)
            }

            return ToolResult.success(result, metadata={
                "operation": "rag_retrieve",
                "top_k": top_k
            })

        except Exception as e:
            return ToolResult.failure(f"RAG retrieval failed: {str(e)}")

    def _mock_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Mock retrieval (replace with real implementation)"""
        # In production, this would query FAISS index
        return [
            {
                "text": f"Mock reference {i+1} for query: {query}",
                "score": 0.9 - i * 0.1,
                "source": f"document_{i+1}"
            }
            for i in range(top_k)
        ]

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema"""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        }


class ToolRegistry:
    """
    Registry for available tools

    Manages tool instances and executes tool calls from JSON format
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default HAVOC tools"""
        self.register(PythonMathTool())
        self.register(DSLExecutorTool())
        self.register(RAGHelperTool())

    def register(self, tool: BaseTool):
        """Register a new tool"""
        self.tools[tool.name] = tool

    def unregister(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with arguments

        Args:
            tool_name: Name of the tool to execute
            args: Dictionary of arguments

        Returns:
            ToolResult object
        """
        tool = self.get_tool(tool_name)

        if tool is None:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error=f"Tool '{tool_name}' not found. Available tools: {self.list_tools()}"
            )

        try:
            return tool.execute(**args)
        except Exception as e:
            return ToolResult.failure(f"Tool execution error: {str(e)}")

    def execute_from_json(self, tool_call_json: str) -> ToolResult:
        """
        Execute a tool from JSON string format: {"tool": "<name>", "args": {...}}

        Args:
            tool_call_json: JSON string specifying tool and args

        Returns:
            ToolResult object
        """
        try:
            tool_call = json.loads(tool_call_json)
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})

            if not tool_name:
                return ToolResult.failure("Missing 'tool' field in JSON")

            return self.execute(tool_name, args)

        except json.JSONDecodeError as e:
            return ToolResult.failure(f"Invalid JSON: {str(e)}")

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all registered tools"""
        return {
            name: tool.get_schema()
            for name, tool in self.tools.items()
        }


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = ToolRegistry()

    # Example 1: Execute t-test
    print("=" * 60)
    print("Example 1: T-Test")
    print("=" * 60)

    result = registry.execute("python_math", {
        "operation": "t_test",
        "group1": [1.2, 1.5, 1.3, 1.6, 1.4],
        "group2": [2.1, 2.3, 2.0, 2.4, 2.2]
    })

    print(result.to_json())

    # Example 2: Execute Box-Behnken design
    print("\n" + "=" * 60)
    print("Example 2: Box-Behnken Design")
    print("=" * 60)

    result = registry.execute("dsl_executor", {
        "operation": "box_behnken",
        "factors": 3
    })

    print(result.to_json())

    # Example 3: Execute from JSON (simulating model output)
    print("\n" + "=" * 60)
    print("Example 3: Execute from JSON")
    print("=" * 60)

    tool_call_json = '''
    {
        "tool": "python_math",
        "args": {
            "operation": "anova",
            "groups": [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        }
    }
    '''

    result = registry.execute_from_json(tool_call_json)
    print(result.to_json())

    # Example 4: List all tools and schemas
    print("\n" + "=" * 60)
    print("Available Tools")
    print("=" * 60)

    for tool_name in registry.list_tools():
        tool = registry.get_tool(tool_name)
        print(f"\n{tool_name}:")
        print(f"  Description: {tool.description}")
        print(f"  Schema: {json.dumps(tool.get_schema(), indent=4)}")
