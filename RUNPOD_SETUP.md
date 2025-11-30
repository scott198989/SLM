# RunPod Environment Setup Guide

## Quick Fix: Install the Package

The `ModuleNotFoundError: No module named 'havoc_core'` means the Python package isn't installed in your RunPod environment.

### Solution: Install in Editable Mode

```bash
# In RunPod terminal
cd /workspace/SLM

# Install the package in editable mode with inference dependencies
pip install -e ".[inference]"

# Or just core dependencies (for training only)
pip install -e .
```

**What this does:**
- `-e` = editable mode (changes to code take effect immediately)
- `.` = install from current directory (uses `pyproject.toml`)
- `[inference]` = optional dependencies for inference server

### After Installation

Verify the package is installed:

```bash
python -c "from havoc_core.config import HavocConfig; print('✅ havoc_core installed successfully')"
```

Then run training:

```bash
python scripts/train.py --config configs/training/havoc_phase1_sft_3b.yaml --max-steps 10
```

## Full RunPod Environment Setup (First Time)

If you're setting up a fresh RunPod instance:

### 1. Clone Repository

```bash
cd /workspace
git clone https://github.com/scott198989/SLM.git
cd SLM
```

### 2. Install System Dependencies (if needed)

```bash
# Update package list
apt-get update

# Install build essentials (usually pre-installed on RunPod)
apt-get install -y build-essential
```

### 3. Install Python Package

```bash
# Install with all dependencies
pip install -e ".[inference,rag]"
```

### 4. Verify PyTorch & CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.1.0+cu121 (or similar)
CUDA Available: True
CUDA Version: 12.1 (or similar)
```

### 5. Check Tokenizer

```bash
ls -la artifacts/tokenizer/
```

If missing, you'll need to train it:
```bash
python -m havoc_core.tokenizer.train_tokenizer
```

### 6. Verify Data Files

```bash
ls -la sft_data/*.jsonl
```

Should show:
- `havoc_sft_phase1_full.jsonl` (38KB)
- Plus chunk files

### 7. Test Training (10 steps)

```bash
python scripts/train.py --config configs/training/havoc_phase1_sft_3b.yaml --max-steps 10
```

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'havoc_core'`
**Solution:** Run `pip install -e .` from `/workspace/SLM`

### Issue: `FileNotFoundError: tokenizer_path ... not found`
**Solution:**
```bash
# Train tokenizer (if you have training data in data/ directory)
python -m havoc_core.tokenizer.train_tokenizer

# Or use a pre-trained tokenizer if you have one
```

### Issue: `No data sources found`
**Solution:** Ensure `sft_data/havoc_sft_phase1_full.jsonl` exists and config specifies `type: jsonl`

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size in config:
```yaml
batch_size: 1  # Already at minimum
gradient_accumulation_steps: 8  # Increase this to maintain effective batch size
```

### Issue: Slow Training
**Solution:**
```bash
# Enable mixed precision (should already be enabled in config)
# Verify in config:
use_amp: true
amp_dtype: bfloat16  # Use bfloat16 on A100, float16 on other GPUs
```

## Recommended RunPod GPU

For HAVOC 3B (2.05B parameters):
- **Minimum:** RTX 3090 (24GB) - batch_size=1, grad_accum=4
- **Recommended:** RTX 4090 (24GB) - batch_size=2, grad_accum=2
- **Optimal:** A100 40GB - batch_size=4, grad_accum=1

Current config uses:
- `batch_size: 1`
- `gradient_accumulation_steps: 4`
- Effective batch size: 4

## Environment Variables (Optional)

```bash
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set visible GPUs (if multi-GPU system)
export CUDA_VISIBLE_DEVICES=0

# Increase file descriptor limit (for large datasets)
ulimit -n 4096
```

## Persistence Between RunPod Sessions

RunPod instances may reset. To persist your work:

### Option 1: Network Volume (Recommended)
- Attach a network volume to `/workspace`
- All files persist across pod restarts

### Option 2: Git Commits
```bash
# Before stopping pod
cd /workspace/SLM
git add -A
git commit -m "Training checkpoint - step 1500"
git push origin main

# After restarting pod
cd /workspace/SLM
git pull origin main
pip install -e .
```

### Option 3: Save Checkpoints to Cloud Storage
```bash
# Install rclone or use AWS S3
# Upload checkpoints after each training session
```

## Quick Reference Commands

```bash
# Install package
pip install -e .

# Train (short test)
python scripts/train.py --config configs/training/havoc_phase1_sft_3b.yaml --max-steps 10

# Train (full)
python scripts/train.py --config configs/training/havoc_phase1_sft_3b.yaml

# Resume from checkpoint
python scripts/train.py --config configs/training/havoc_phase1_sft_3b.yaml --resume checkpoints/havoc_sft_phase1/checkpoint_step_500

# Monitor GPU usage
nvidia-smi -l 1  # Update every 1 second

# Check training logs
tail -f logs/train.log
```

## Next Steps After Training

1. **Start Inference Server:**
```bash
python scripts/serve.py --checkpoint checkpoints/havoc_sft_phase1/checkpoint_step_1500
```

2. **Test Generation:**
```bash
curl -X POST http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is sodium borohydride?", "max_new_tokens": 100}'
```

3. **Run Frontend (requires port forwarding or RunPod public URL):**
```bash
cd frontend
npm install
npm run dev
```
