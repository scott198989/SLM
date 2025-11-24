#!/usr/bin/env python3
"""
Test script to verify all imports resolve correctly.

This script attempts to import all major modules and reports any failures.
"""

import sys
from typing import List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def test_import(module_name: str, items: List[str] = None) -> Tuple[bool, str]:
    """Test importing a module or specific items from it."""
    try:
        if items:
            exec(f"from {module_name} import {', '.join(items)}")
            return True, f"from {module_name} import {', '.join(items)}"
        else:
            exec(f"import {module_name}")
            return True, f"import {module_name}"
    except Exception as e:
        error_msg = str(e).split('\n')[0]  # Get first line of error
        return False, f"{module_name}: {error_msg}"


def main():
    print(f"{BOLD}=== HAVOC-7B Import Verification ==={RESET}\n")

    tests = [
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

    passed = 0
    failed = 0
    failures = []

    for module, items in tests:
        success, msg = test_import(module, items)
        if success:
            print(f"{GREEN}✓{RESET} {msg}")
            passed += 1
        else:
            print(f"{RED}✗{RESET} {msg}")
            failed += 1
            failures.append(msg)

    print(f"\n{BOLD}=== Summary ==={RESET}")
    print(f"Passed: {GREEN}{passed}{RESET}")
    print(f"Failed: {RED}{failed}{RESET}")

    if failures:
        print(f"\n{BOLD}Failures:{RESET}")
        for failure in failures:
            print(f"  • {failure}")
        return 1
    else:
        print(f"\n{GREEN}{BOLD}All imports successful!{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
