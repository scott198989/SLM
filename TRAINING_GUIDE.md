# HAVOC-7B Training Guide

## Overview

The training stack is now complete! You can train the HAVOC-7B model from scratch with a simple command:

```bash
python scripts/train.py --config configs/training/default_training.yaml
```

## What's Been Implemented

### 1. **TrainingConfig** (`src/havoc_core/config.py`)
Comprehensive training configuration with:
- Hyperparameters (batch size, learning rate, weight decay, etc.)
- Mixed precision training (AMP with bfloat16/float16)
- Learning rate scheduling (cosine, linear, constant)
- Gradient accumulation and clipping
- Checkpoint management
- Validation settings
- Logging configuration

### 2. **Trainer Class** (`src/havoc_training/trainer.py`)
Full-featured trainer with:
- ✅ **Optimizer**: AdamW with weight decay (excluding bias/LayerNorm)
- ✅ **Scheduler**: Configurable LR schedule with warmup (cosine/linear/constant)
- ✅ **Mixed Precision**: torch.cuda.amp with gradient scaling
- ✅ **Gradient Accumulation**: Effective batch size control
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Training Loop**: Epoch-based or step-based training
- ✅ **Validation Loop**: Perplexity calculation on held-out data
- ✅ **Checkpointing**: Save/load/resume with automatic cleanup
- ✅ **Logging**: Console + file logging with configurable frequency
- ✅ **Reproducibility**: Random seed setting across numpy/torch/cuda

### 3. **Training Script** (`scripts/train.py`)
CLI entrypoint with:
- YAML config loading
- Command-line overrides for key parameters
- Automatic dataset creation (with warnings if no data present)
- Dummy dataset support for testing
- Graceful error handling (saves checkpoint on interrupt/error)
- Resume from checkpoint support

### 4. **Default Config** (`configs/training/default_training.yaml`)
Production-ready configuration with:
- 7B parameter model setup
- Data mixture ratios (60% domain, 30% general, 10% dialog)
- Reasonable hyperparameters for single-GPU training
- Hardware-specific recommendations for different GPUs
- Detailed comments explaining each setting

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

This installs:
- PyTorch >= 2.1
- NumPy, SciPy, SymPy
- SentencePiece (for tokenizer)
- PyYAML (for configs)
- Optional: faiss-cpu (for RAG)

### 2. Prepare Your Data

Create a `data/` directory and organize your text files:

```
data/
├── math/
│   ├── calculus.txt
│   ├── linear_algebra.txt
│   └── differential_equations.txt
├── stats/
│   ├── probability.txt
│   ├── statistical_inference.txt
│   └── regression.txt
├── engineering/
│   ├── doe_design_of_experiments.txt
│   ├── six_sigma_process.txt
│   └── materials_science.txt
└── general/
    ├── reasoning.txt
    └── world_knowledge.txt
```

**Important**: The training script will work with dummy data if no data directory exists, but you'll want real data for actual training!

### 3. Train the Model

Basic training:
```bash
python scripts/train.py
```

With custom config:
```bash
python scripts/train.py --config configs/training/default_training.yaml
```

Override specific parameters:
```bash
python scripts/train.py \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-steps 10000 \
  --device cuda
```

Resume from checkpoint:
```bash
python scripts/train.py --resume checkpoints/checkpoint_step_5000
```

### 4. Monitor Training

Logs are saved to:
- **Console**: Real-time training progress
- **File**: `logs/train.log` (detailed logs)
- **Checkpoints**: `checkpoints/checkpoint_step_N/`

Training metrics logged:
- Training loss (every 10 steps by default)
- Learning rate (every 10 steps)
- Validation loss (every 500 steps)
- Validation perplexity (every 500 steps)

## Configuration Tips

### For Different Hardware

**RTX 3090 / RTX 4090 (24GB)**:
```yaml
batch_size: 2
gradient_accumulation_steps: 16
use_amp: true
amp_dtype: bfloat16
```

**A100 40GB**:
```yaml
batch_size: 4
gradient_accumulation_steps: 8
use_amp: true
amp_dtype: bfloat16
```

**A100 80GB**:
```yaml
batch_size: 8
gradient_accumulation_steps: 4
use_amp: true
amp_dtype: bfloat16
```

**CPU (testing only)**:
```yaml
batch_size: 1
gradient_accumulation_steps: 1
use_amp: false
device: cpu
```

### Recommended Hyperparameters

Based on modern LLM training best practices:

- **Learning Rate**: 3e-4 (peak), 3e-5 (min)
- **Weight Decay**: 0.1
- **Warmup Steps**: 2000
- **Gradient Clipping**: 1.0
- **LR Schedule**: Cosine with warmup
- **Optimizer**: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- **Mixed Precision**: bfloat16 (preferred) or float16

## Checkpoint Management

Checkpoints are saved every 1000 steps by default. Each checkpoint contains:
- `model.pt` - Model state dict
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - LR scheduler state
- `scaler.pt` - AMP gradient scaler state (if using AMP)
- `training_state.json` - Global step, epoch, best val loss
- `config.json` - Full training configuration

Only the last 3 checkpoints are kept by default to save disk space.

## Validation & Perplexity

Validation runs every 500 steps on 100 batches by default. Metrics:
- **Loss**: Cross-entropy loss on held-out data
- **Perplexity**: exp(loss) - lower is better

Good perplexity targets:
- **Domain data** (math/stats): 10-30
- **General data**: 30-50
- **Mixed**: 20-40

## Interrupting Training

Training can be safely interrupted:
- **Ctrl+C**: Saves `checkpoint_interrupted`
- **Error/Crash**: Saves `checkpoint_error` (if possible)

Resume with:
```bash
python scripts/train.py --resume checkpoints/checkpoint_interrupted
```

## Next Steps

Once training is complete:
1. **Evaluate**: Use `havoc_eval` benchmarks to test performance
2. **Fine-tune**: Use the same script with `--resume` for fine-tuning
3. **Serve**: Deploy with the inference API (coming soon)
4. **RAG**: Integrate with `havoc_rag` for retrieval-augmented generation

## Troubleshooting

**Out of Memory (OOM)**:
- Reduce `batch_size`
- Increase `gradient_accumulation_steps` (keeps effective batch size same)
- Enable mixed precision (`use_amp: true`)
- Reduce `max_seq_len` in model config

**Slow Training**:
- Enable mixed precision
- Use bfloat16 on A100/H100 GPUs
- Increase batch size if memory allows
- Use compiled model (PyTorch 2.0+): Add `torch.compile()` to model

**NaN Loss**:
- Reduce learning rate
- Check data for corrupted examples
- Enable gradient clipping
- Use bfloat16 instead of float16

**Training Not Converging**:
- Increase warmup steps
- Reduce learning rate
- Check data quality and mixture
- Verify tokenization is correct

## Philosophy: No Sycophancy

This training stack is **production-ready** but **not magic**. Real considerations:

1. **Training a 7B model from scratch is expensive**:
   - ~100B tokens minimum for reasonable performance
   - ~1000 A100 GPU-hours
   - Requires careful data curation

2. **You'll need real data**:
   - The dummy dataset is just for testing
   - Quality > Quantity for domain specialization
   - Deduplication and cleaning are critical

3. **Hyperparameters matter**:
   - Don't just use defaults blindly
   - Monitor validation loss closely
   - Adjust based on your data characteristics

4. **Hardware constraints are real**:
   - Single GPU training is slow but possible
   - Multi-GPU support can be added (not included yet)
   - Consider cloud GPUs or clusters for serious training

## What's Still Missing

For true "press train and go", you might want:
- **Multi-GPU training** (DDP, FSDP)
- **Data pipeline improvements** (streaming, preprocessing)
- **Trained tokenizer** (currently uses dummy)
- **Experiment tracking** (Weights & Biases, TensorBoard)
- **Automatic hyperparameter tuning**

These can be added incrementally as needed.

---

**Ready to train? Just choose your datasets and run:**
```bash
python scripts/train.py
```
