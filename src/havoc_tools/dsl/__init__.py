"""DSL (Domain-Specific Language) Module

Parser and executor for HAVOC's domain-specific language for statistical operations.
"""

from havoc_tools.dsl.executor import DSLExecutor, ExecutionOutcome
from havoc_tools.dsl.parser import parse_dsl
from havoc_tools.dsl.spec import DSLRequest, DesignDOE, EvalSPC, Factor

__all__ = [
    "DSLExecutor",
    "ExecutionOutcome",
    "parse_dsl",
    "DSLRequest",
    "DesignDOE",
    "EvalSPC",
    "Factor",
]
