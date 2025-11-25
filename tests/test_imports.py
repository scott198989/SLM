from __future__ import annotations

import importlib
from typing import Iterable, Sequence

import pytest

TESTS: Sequence[tuple[str, Iterable[str] | None]] = [
    # Core modules
    ("havoc_core", None),
    ("havoc_core.config", ["HavocConfig", "TrainingConfig", "InferenceConfig"]),
    ("havoc_core.model.transformer", ["HavocModel"]),
    ("havoc_core.model.blocks", ["RMSNorm", "TransformerBlock"]),
    # Data module
    ("havoc_data", None),
    ("havoc_data.dataset", ["CausalLMDataset"]),
    ("havoc_data.sources", ["DataSource"]),
    ("havoc_data.preprocess", ["normalize_text"]),
    # Training module
    ("havoc_training", None),
    ("havoc_training.trainer", ["Trainer"]),
    # Inference module
    ("havoc_inference", None),
    ("havoc_inference.engine", ["InferenceEngine"]),
    ("havoc_inference.server", ["create_app"]),
    # SRS module
    ("havoc_srs", None),
    ("havoc_srs.orchestrator", ["run_pipeline"]),
    ("havoc_srs.answer", ["Answer"]),
    ("havoc_srs.arbiter", ["decide"]),
    # RAG module
    ("havoc_rag", None),
    ("havoc_rag.retrieval", ["Retriever"]),
    ("havoc_rag.embeddings", ["EmbeddingModel"]),
    ("havoc_rag.index", ["VectorIndex"]),
    # Tools module
    ("havoc_tools", None),
    ("havoc_tools.dsl.parser", ["parse_dsl"]),
    ("havoc_tools.dsl.executor", ["DSLExecutor"]),
    ("havoc_tools.python_math.engine", None),
    # Eval module
    ("havoc_eval", None),
    ("havoc_eval.harness", ["run_eval"]),
    ("havoc_eval.benchmarks", ["default_benchmarks"]),
    # CLI module
    ("havoc_cli", None),
    ("havoc_cli.main", ["main"]),
]


@pytest.mark.parametrize("module_name, items", TESTS)
def test_imports(module_name: str, items: Iterable[str] | None) -> None:
    """Ensure modules and key symbols import successfully."""
    module = importlib.import_module(module_name)
    if items:
        for item in items:
            assert hasattr(module, item), f"{module_name} missing {item}"
