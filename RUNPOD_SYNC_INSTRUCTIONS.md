# RunPod Environment Sync Instructions

## Problem
Your RunPod instance has uncommitted changes that conflict with the latest fixes pushed to GitHub (commit 66bfdf1).

## Solution: Sync RunPod with Latest Code

### Option 1: Stash and Pull (Recommended - Preserves Local Changes)

```bash
# In RunPod terminal
cd /workspace/SLM

# Save your local changes
git stash save "RunPod local changes before sync"

# Pull latest code from main branch
git pull origin main

# Re-apply your local changes
git stash pop

# If there are conflicts, resolve them manually
# Then continue with training
```

### Option 2: Commit Local Changes First

```bash
# In RunPod terminal
cd /workspace/SLM

# Review what changed
git diff configs/training/havoc_phase1_sft_3b.yaml

# Commit your changes
git add configs/training/havoc_phase1_sft_3b.yaml
git commit -m "RunPod local config changes"

# Pull latest (may create merge commit)
git pull origin main

# Resolve any conflicts if needed
```

### Option 3: Discard Local Changes (Clean Pull)

**WARNING:** This will lose any uncommitted changes in RunPod!

```bash
# In RunPod terminal
cd /workspace/SLM

# Discard all local changes
git reset --hard HEAD

# Pull latest code
git pull origin main
```

## What Changed in Latest Commit (66bfdf1)

1. **RMSNorm Fix** - Fixed string-to-float conversion bug in `src/havoc_core/model/blocks.py:19`
2. **Config Updates** - Updates to training configuration files
3. **Type Safety** - Enhanced YAML numeric parameter handling (now in train.py)

## After Syncing: Push Your Local Changes Back to GitHub

If you want to sync your local Windows environment with RunPod changes:

```bash
# On Windows (C:\Users\ScottT\SLM)
git pull origin main
```

This will pull both the latest commit AND any changes you committed in RunPod.

## Verification

After syncing, verify you're on the latest code:

```bash
# In RunPod
cd /workspace/SLM
git log -1 --oneline

# Should show commit 66bfdf1 or newer
# Output: 66bfdf1 ...
```

Then test training:

```bash
python scripts/train.py --config configs/training/havoc_phase1_sft_3b.yaml --max-steps 10
```

## Expected Output

After fixes, you should see:
- ✅ No TypeError about 'Tensor' and 'str'
- ✅ JSONL data loads correctly
- ✅ Training progresses through 10 steps
- ✅ Loss values displayed

## Troubleshooting

### If git stash fails
```bash
# Create a backup branch instead
git checkout -b runpod-backup
git add -A
git commit -m "Backup before sync"
git checkout main
git pull origin main
```

### If merge conflicts occur
```bash
# For config file conflicts, you can choose one version:
git checkout --ours configs/training/havoc_phase1_sft_3b.yaml   # Keep RunPod version
# OR
git checkout --theirs configs/training/havoc_phase1_sft_3b.yaml # Use GitHub version

# Then continue
git add configs/training/havoc_phase1_sft_3b.yaml
git commit -m "Resolved merge conflict"
```

### If tokenizer is missing
```bash
ls -la /workspace/SLM/artifacts/tokenizer/

# If missing, train tokenizer first:
python -m havoc_core.tokenizer.train_tokenizer
```

### If sft_data files are missing
```bash
ls -la /workspace/SLM/sft_data/*.jsonl

# Files should be present from your Windows host
# If not, you may need to upload them to RunPod
```
