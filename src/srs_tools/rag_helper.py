"""
RAG Helper - Retrieval-Augmented Generation tool for HAVOC PRIME

Wraps existing RAG retrieval with a clean interface.
"""

from __future__ import annotations

from typing import Any, Dict, List

from havoc_rag.retrieval import Retriever


class RAGHelper:
    """
    Callable interface to RAG retrieval.

    Methods:
        retrieve: Get relevant references for a query
        add_documents: Add documents to the index
    """

    def __init__(self, embed_dim: int = 384):
        self.retriever = Retriever(embed_dim=embed_dim)
        self._initialized = False

    def initialize_corpus(self, documents: List[str]) -> None:
        """
        Initialize corpus with documents.

        Args:
            documents: List of text documents
        """
        self.retriever.add_corpus(documents)
        self._initialized = True

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant references for query.

        Args:
            query: Search query
            k: Number of references to retrieve

        Returns:
            List of reference dicts with 'text' and 'score'
        """
        if not self._initialized:
            # Use default corpus if not initialized
            self._initialize_default_corpus()

        references = self.retriever.retrieve_references(query, k=k)
        return references

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to corpus.

        Args:
            documents: List of text documents
        """
        self.retriever.add_corpus(documents)

    def _initialize_default_corpus(self) -> None:
        """Initialize with default domain knowledge"""
        default_docs = [
            "ANOVA (Analysis of Variance) explains variance components and tests mean differences across multiple groups. It partitions total variance into between-group and within-group components.",
            "Design of Experiments (DOE) helps understand factor interactions and optimize processes. Common designs include full factorial, fractional factorial, Box-Behnken, and central composite designs.",
            "Statistical Process Control (SPC) monitors process stability using control charts and statistical rules. Common charts include I-MR, Xbar-R, and p-charts.",
            "T-tests compare means between two groups with assumptions of normality and independence. Welch's t-test relaxes the equal variance assumption.",
            "Linear regression models relationships between variables and predicts outcomes. Key metrics include R-squared, adjusted R-squared, and residual analysis.",
            "Western Electric rules detect special cause variation in control charts. Rules include points beyond control limits, runs, trends, and patterns.",
            "Factorial designs explore all combinations of factor levels to identify main effects and interactions. 2^k designs are common for k factors at 2 levels.",
            "P-values indicate statistical significance but should be interpreted with effect size and practical significance. A p-value < 0.05 suggests rejecting the null hypothesis.",
            "Box-Behnken designs are response surface designs that require 3 levels per factor and are more efficient than full factorial designs for optimization.",
            "Process capability indices (Cp, Cpk) measure how well a process meets specifications. Cpk accounts for process centering while Cp assumes perfect centering.",
            "Control limits are typically set at ±3 sigma from the center line. These limits represent the expected natural variation of a stable process.",
            "Interaction effects occur when the effect of one factor depends on the level of another factor. These are revealed through interaction plots in DOE.",
            "Regression assumptions include linearity, independence, homoscedasticity (equal variance), and normality of residuals. Violations can invalidate results.",
            "Sample size determination depends on desired power, effect size, and significance level. Larger samples detect smaller effects but cost more.",
            "Confidence intervals provide a range of plausible values for a parameter. A 95% CI means the interval would contain the true parameter in 95% of repeated samples."
        ]

        self.retriever.add_corpus(default_docs)
        self._initialized = True
