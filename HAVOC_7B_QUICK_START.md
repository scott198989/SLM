# HAVOC-7B + PRIME: Quick Start Guide

**Version:** 2.0
**Last Updated:** December 2, 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Training Pipeline](#training-pipeline)
5. [Usage Examples](#usage-examples)
6. [File Reference](#file-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

HAVOC-7B is a **7 billion parameter decoder-only transformer** with integrated **PRIME meta-reasoning**. This guide will walk you through training and using the model.

### What You'll Get

- ✅ Complete 7B parameter model with GQA, RoPE, SwiGLU
- ✅ PRIME meta-reasoning (budget-based, adversarial)
- ✅ Visible chain-of-thought with reasoning tokens
- ✅ Tool-calling support (Python math, DSL, RAG)
- ✅ Optimized for single-GPU training (RTX 5090 24GB)
- ✅ Production-ready training pipeline

---

## System Requirements

### Minimum Requirements (Training)

- **GPU:** RTX 5090 (24GB VRAM) or equivalent
- **RAM:** 64GB system RAM
- **Storage:** 500GB SSD (for checkpoints and data)
- **OS:** Linux (Ubuntu 22.04+ recommended) or Windows 11

### Recommended Requirements

- **GPU:** RTX 5090 or A100 40GB
- **RAM:** 128GB system RAM
- **Storage:** 1TB NVMe SSD
- **OS:** Ubuntu 22.04 LTS

### Software Dependencies

```bash
Python >= 3.10
PyTorch >= 2.1 (with CUDA 12.1)
sentencepiece >= 0.1.99
scipy >= 1.11
numpy >= 1.26
```

---

## Installation

### Step 1: Clone Repository

```bash
cd /workspace/SLM
# Repository already exists
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentencepiece scipy numpy pyyaml
pip install -e .
```

### Step 3: Verify Installation

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from havoc_core.config_7b import Havoc7BConfig; print(Havoc7BConfig())"
```

---

## Training Pipeline

The training pipeline consists of **4 phases**:

```
Phase 0: Tokenizer Training (1-2 hours)
    ↓
Phase 1: Pretraining (1000 GPU-hours / ~42 days on RTX 5090)
    ↓
Phase 2: Supervised Fine-Tuning (100 GPU-hours / ~4 days)
    ↓
Phase 3: Conversational Polish (50 GPU-hours / ~2 days)
```

### Phase 0: Train Tokenizer

**Goal:** Build 70k vocabulary tokenizer with reasoning tokens

```bash
# Prepare corpus (10GB+ text)
mkdir -p data/corpus
# Add your text files to data/corpus/

# Train tokenizer
python scripts/phase0_train_tokenizer.py \
    --input-dir data/corpus \
    --output-dir artifacts/tokenizer \
    --vocab-size 70000

# Output: artifacts/tokenizer/tokenizer.model
```

**Time:** 1-2 hours
**Output:** `artifacts/tokenizer/tokenizer.model`, `tokenizer.vocab`

### Phase 1: Pretrain Base Model

**Goal:** Train base language model on 100B tokens

**Data Preparation:**

```bash
# Organize data by domain
mkdir -p data/{math,stats,engineering,general,code}

# Expected data mixture:
# - math: 25GB (25% of training)
# - stats: 20GB (20%)
# - engineering: 15GB (15%)
# - general: 30GB (30%)
# - code: 10GB (10%)
# Total: ~100GB of text
```

**Run Training:**

```bash
python scripts/phase1_pretrain_7b.py \
    --data-dir data \
    --tokenizer-path artifacts/tokenizer \
    --checkpoint-dir checkpoints/phase1 \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --max-steps 100000 \
    --max-seq-len 2048

# Monitor training
tail -f logs/havoc_7b/train.log
```

**Time:** ~1000 GPU-hours (42 days on RTX 5090)
**Output:** `checkpoints/phase1/checkpoint_step_100000/`

**Memory Usage:**

```
Model (bf16):           14 GB
Gradients (bf16):       14 GB
Activations (gradient checkpointing): 2 GB
Flash Attention:         1 GB
Misc buffers:            1 GB
─────────────────────────────
TOTAL:                  ~22 GB (fits in 24GB!)
```

### Phase 2: Supervised Fine-Tuning (SFT)

**Goal:** Teach reasoning, tool use, and PRIME

**Data Preparation:**

Create SFT examples in JSONL format:

```jsonl
{"prompt": "Design a Box-Behnken DOE for 3 factors", "completion": "<reason>\nTask: Design Box-Behnken DOE...\n</reason>\n<tool>\n{\"tool\": \"dsl_executor\", \"args\": {\"operation\": \"box_behnken\", \"factors\": 3}}\n</tool>\n\nFinal answer: ..."}
{"prompt": "Compare two groups using t-test", "completion": "<reason>\nSubgoal 1: Extract data...\n</reason>\n<tool>\n{\"tool\": \"python_math\", ...}\n</tool>\n..."}
```

Save to `data/sft/examples.jsonl` (aim for 100k examples)

**Run SFT:**

```bash
python scripts/phase2_sft_7b.py \
    --checkpoint checkpoints/phase1/checkpoint_step_100000 \
    --sft-data data/sft \
    --output-dir checkpoints/phase2 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-steps 10000 \
    --learning-rate 1e-5

# Time: ~100 GPU-hours (~4 days)
```

**Output:** `checkpoints/phase2/checkpoint_step_10000/`

### Phase 3: Conversational Polish

**Goal:** Polish conversational quality and safety

**Data Preparation:**

Create conversational examples:

```jsonl
{"messages": [{"role": "user", "content": "Explain t-test"}, {"role": "assistant", "content": "A t-test is a statistical test..."}]}
```

Save to `data/polish/conversations.jsonl` (50k examples)

**Run Polish:**

```bash
python scripts/phase3_polish_7b.py \
    --checkpoint checkpoints/phase2/checkpoint_step_10000 \
    --polish-data data/polish \
    --output-dir checkpoints/phase3 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-steps 5000 \
    --learning-rate 5e-6

# Time: ~50 GPU-hours (~2 days)
```

**Output:** `checkpoints/phase3/checkpoint_step_5000/` ← **Final model!**

---

## Usage Examples

### Example 1: Load Model and Generate

```python
from havoc_core.model.prime_model import HavocPrimeModel
import sentencepiece as spm

# Load model
model = HavocPrimeModel.from_pretrained(
    "checkpoints/phase3/checkpoint_step_5000",
    device="cuda"
)

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("artifacts/tokenizer/tokenizer.model")

# Generate (simple, no PRIME)
result = model.generate_with_prime(
    prompt="What is 2 + 2?",
    tokenizer=tokenizer,
    enable_prime=False,
    max_new_tokens=50
)

print(result.text)
# Output: "2 + 2 = 4"
```

### Example 2: PRIME Reasoning (MEDIUM Budget)

```python
# Stats problem with reasoning
result = model.generate_with_prime(
    prompt="Compare groups [1,2,3,4,5] and [2,3,4,5,6] using t-test",
    tokenizer=tokenizer,
    enable_prime=True,
    max_new_tokens=512
)

print(result.text)
# Output:
# <reason>
# Task: Compare two groups using t-test
# Subgoal 1: Extract data from prompt
# Subgoal 2: Run t-test tool
# Subgoal 3: Interpret results at α=0.05
# </reason>
#
# <tool>
# {"tool": "python_math", "args": {"operation": "t_test", "group1": [1,2,3,4,5], "group2": [2,3,4,5,6]}}
# </tool>
#
# Tool result: {"status": "success", "result": {"statistic": -2.236, "pvalue": 0.048, ...}}
#
# <advocate>
# The p-value (0.048) is less than 0.05, indicating statistical significance.
# </advocate>
#
# <attack>
# However, the effect size is small and samples sizes are limited (n=5).
# </attack>
#
# <pragmatist>
# While statistically significant, practical significance should be evaluated.
# </pragmatist>
#
# Final answer: The groups differ significantly (p=0.048), but consider collecting more data for stronger conclusions.

# Access structured results
print(f"Routing: {result.routing.budget}")  # MEDIUM
print(f"Tool calls: {len(result.tool_results)}")  # 1
print(f"Verification passed: {result.verification['passed']}")  # True
```

### Example 3: Full PRIME (HEAVY Budget)

```python
# Complex DOE design
result = model.generate_with_prime(
    prompt="Design a Box-Behnken DOE for temperature (100-200°C), pressure (10-50 bar), speed (50-150 RPM)",
    tokenizer=tokenizer,
    enable_prime=True,
    max_new_tokens=1024
)

print(result.text)
# Full PRIME reasoning with:
# - Multiple subgoals
# - Tool calls to DSL executor
# - Adversarial reasoning (advocate/attack/pragmatist)
# - Verification
# - Compressed final answer

# Workspace summary
print(result.workspace.summarize())
# {
#   "facts_count": 12,
#   "constraints_count": 3,
#   "partial_results": 4,
#   "global_confidence": 0.89
# }
```

### Example 4: Tool-Only Interface

```python
from havoc_core.tool_interface import ToolRegistry

# Create tool registry
tools = ToolRegistry()

# Execute tool directly
result = tools.execute("python_math", {
    "operation": "anova",
    "groups": [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
})

print(result.to_json())
# {
#   "status": "success",
#   "result": {
#     "f_statistic": 6.0,
#     "pvalue": 0.028,
#     "significant": true
#   }
# }
```

---

## File Reference

### Core Architecture Files

| File | Description |
|------|-------------|
| [`src/havoc_core/config_7b.py`](src/havoc_core/config_7b.py) | 7B model configuration |
| [`src/havoc_core/model/prime_model.py`](src/havoc_core/model/prime_model.py) | PRIME-integrated model |
| [`src/havoc_core/reasoning_tokens.py`](src/havoc_core/reasoning_tokens.py) | Reasoning token system |
| [`src/havoc_core/tool_interface.py`](src/havoc_core/tool_interface.py) | Tool registry and execution |

### Training Files

| File | Description |
|------|-------------|
| [`src/havoc_training/optimized_trainer.py`](src/havoc_training/optimized_trainer.py) | Memory-optimized trainer |
| [`scripts/phase0_train_tokenizer.py`](scripts/phase0_train_tokenizer.py) | Tokenizer training |
| [`scripts/phase1_pretrain_7b.py`](scripts/phase1_pretrain_7b.py) | Pretraining script |
| [`scripts/phase2_sft_7b.py`](scripts/phase2_sft_7b.py) | SFT script |
| [`scripts/phase3_polish_7b.py`](scripts/phase3_polish_7b.py) | Polish script |

### Documentation

| File | Description |
|------|-------------|
| [`HAVOC_7B_PRIME_ARCHITECTURE.md`](HAVOC_7B_PRIME_ARCHITECTURE.md) | Complete architecture spec |
| [`HAVOC_7B_QUICK_START.md`](HAVOC_7B_QUICK_START.md) | This file |

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms:** CUDA OOM error during training

**Solutions:**

1. **Reduce sequence length:**
   ```bash
   --max-seq-len 1024  # Instead of 2048
   ```

2. **Enable CPU offloading:**
   ```python
   train_config.cpu_offload_optimizer = True
   ```

3. **Increase gradient checkpointing frequency:**
   ```python
   train_config.checkpoint_every_n_layers = 2  # Checkpoint every 2 layers
   ```

### Issue: Slow Training

**Symptoms:** < 1 step/second

**Solutions:**

1. **Verify Flash Attention is enabled:**
   ```python
   assert train_config.use_flash_attention == True
   ```

2. **Check mixed precision:**
   ```python
   assert train_config.use_amp == True
   assert train_config.amp_dtype == "bfloat16"
   ```

3. **Use fused AdamW:**
   ```python
   train_config.optimizer = "adamw_fused"
   ```

### Issue: NaN Loss

**Symptoms:** Loss becomes NaN during training

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   --learning-rate 1e-4  # Lower LR
   ```

2. **Use bfloat16 instead of float16:**
   ```python
   train_config.amp_dtype = "bfloat16"  # More stable
   ```

3. **Enable gradient clipping:**
   ```python
   train_config.max_grad_norm = 1.0  # Already default
   ```

### Issue: Reasoning Tokens Not Generated

**Symptoms:** Model doesn't generate `<reason>...</reason>`

**Solutions:**

1. **Verify SFT data includes reasoning examples:**
   ```bash
   grep "<reason>" data/sft/*.jsonl
   ```

2. **Check tokenizer includes reasoning tokens:**
   ```python
   tokenizer.encode("<reason>")  # Should return valid token ID
   ```

3. **Increase SFT training steps:**
   ```bash
   --max-steps 20000  # More training
   ```

---

## Next Steps

1. **Prepare Training Data:** Organize 100GB+ corpus for Phase 1
2. **Train Tokenizer:** Run Phase 0 (1-2 hours)
3. **Start Pretraining:** Run Phase 1 (expect 42 days)
4. **Monitor Training:** Check logs regularly
5. **Fine-Tune:** Run Phases 2 and 3 after pretraining
6. **Evaluate:** Test model on domain-specific tasks

---

## Support

For issues or questions:

1. Check [`HAVOC_7B_PRIME_ARCHITECTURE.md`](HAVOC_7B_PRIME_ARCHITECTURE.md) for detailed architecture info
2. Review [`CLAUDE.md`](CLAUDE.md) for development guidelines
3. Check training logs in `logs/havoc_7b/`

---

**Last Updated:** December 2, 2025
**Version:** 2.0
