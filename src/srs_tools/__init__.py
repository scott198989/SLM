"""
SRS Tools - Domain-specific toolbox for HAVOC PRIME

This package contains isolated, callable tools for:
- DSL execution (DOE/SPC/STATS)
- Python math/stats operations
- RAG retrieval
- Answer formatting

These tools are called BY PRIME, not as a pipeline.
"""

from srs_tools.answer_formatter import AnswerFormatter
from srs_tools.dsl_executor import DSLExecutor
from srs_tools.python_math import PythonMathEngine
from srs_tools.rag_helper import RAGHelper

__all__ = [
    "DSLExecutor",
    "PythonMathEngine",
    "RAGHelper",
    "AnswerFormatter"
]
