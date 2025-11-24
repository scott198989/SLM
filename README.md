# HAVOC-7B / SIGMA-7B Engineering Base

This repository provides the scaffolding for a domain-specialized ~7B decoder-only transformer plus the Scott Reasoning Stack (SRS-7B) and supporting tooling. It includes model definitions, tokenizer pipeline, data plumbing, math/statistics tools, a RAG layer, and the multi-stage SRS orchestrator. **Training loops and real weights are intentionally out of scope for now.**

## Repository Layout

```
src/
  havoc_core/        # configs, model architecture, tokenizer tooling
  havoc_data/        # data sources, normalization, dataset abstractions
  havoc_tools/       # math/stats engine and DOE/SPC DSL executor
  havoc_rag/         # embedding wrapper, vector index, retrieval API
  havoc_srs/         # MODE→GROUND→PLAN→EXECUTE→ARGUE→ARBITER→AUDIT→ANSWER
  havoc_eval/        # benchmark registry and evaluation harness
  havoc_cli/         # CLI entrypoints
configs/             # sample YAML configs for model, data, tools, RAG, SRS
scripts/             # helper scripts (dev checklist, demo)
tests/               # smoke/unit tests
```

## Core Model (HAVOC-7B)
- Decoder-only transformer: 32 layers, d_model=4096, 32 heads, head_dim=128, SwiGLU MLP (~11008).
- Grouped-Query Attention with RoPE and RMSNorm everywhere.
- Defined in `src/havoc_core/model/transformer.py` and `src/havoc_core/model/blocks.py` with configuration in `src/havoc_core/config.py`.
- Provides forward pass, KV-cache support, simple greedy `generate`, and config serialization hooks.

## Tokenizer Pipeline
- SentencePiece-based trainer (`src/havoc_core/tokenizer/train_tokenizer.py`) with normalization helpers and domain DSL tokens registered via `vocab_utils.py`.
- Target vocab size ~70–80k; reserved tokens for DSL and SRS markers.
- Returns metadata describing special and domain tokens; no trained artifacts committed.

## Data Pipeline
- `havoc_data.sources` describes weighted data sources.
- `havoc_data.preprocess` normalizes whitespace and engineering/math symbols.
- `havoc_data.dataset` offers a PyTorch-style causal LM dataset with padding to configured sequence length.
- Mixture ratios configurable via `DataMixtureConfig` in `havoc_core.config`.

## Tools and DSL
- Math/stats engine (`havoc_tools/python_math/engine.py`) exposes typed helpers for t-tests, ANOVA, regression, DOE analysis, and symbolic derivatives.
- DOE/SPC DSL defined in `havoc_tools/dsl/spec.py`, parsed via `parser.py`, and executed through `executor.py` that maps DSL to engine calls (DOE example wired).

## RAG Layer
- Lightweight embedding wrapper (`havoc_rag/embeddings.py`) and in-memory vector index (`havoc_rag/index.py`).
- Retrieval interface (`havoc_rag/retrieval.py`) supports indexing corpora and top-k lookups for grounding.

## SRS-7B Reasoning Stack
- Explicit stages implemented under `havoc_srs/`: MODE classifier, GROUND retrieval, PLAN builder, EXECUTE tool calls, ARGUE pro/con stubs, ARBITER decision, AUDIT checks, and ANSWER assembly.
- Orchestrator (`havoc_srs/orchestrator.py`) wires stages together for end-to-end dry runs.

## Evaluation Harness
- Benchmark registry (`havoc_eval/benchmarks.py`) and harness (`havoc_eval/harness.py`) to run SRS over smoke tests and capture outputs/confidence.

## CLI
- `python -m havoc_cli.main "PROMPT"` runs the orchestrator against a prompt using default configs.

## Scripts
- `scripts/demo_run.py`: minimal demonstration of the SRS pipeline.
- `scripts/dev_checklist.sh`: placeholder for lint/typecheck/test (extend as needed).

## Getting Started
1. Install dependencies (consider a virtualenv):
   ```bash
   pip install -e .
   ```
2. Run a smoke pipeline:
   ```bash
   python -m havoc_cli.main "Run a two-sample t-test on dataset A vs B"
   ```
3. Train a tokenizer on your text files (corpus paths required):
   ```bash
   python -m havoc_core.tokenizer.train_tokenizer --help  # adjust config inside script or import and call train_tokenizer
   ```

## Notes and Next Steps
- No training loops or weights are included; hooks and TODOs mark where to integrate future training and RAG corpora.
- Swap in real embeddings/indices and production-grade DSL/tool mappings when available.
