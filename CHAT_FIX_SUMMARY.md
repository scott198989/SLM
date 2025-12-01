# Chat Script Configuration Fix Summary

**Date:** December 1, 2025
**Issue:** RuntimeError when loading checkpoint in `chat_havoc.py`
**Status:** ✅ RESOLVED

---

## Problem Description

When running `python chat_havoc.py`, the script crashed with a `RuntimeError` due to size mismatches between the checkpoint weights and the model being instantiated:

```
RuntimeError: Error(s) in loading state_dict for HavocModel:
    size mismatch for layers.0.attn.q_proj.weight:
        copying a param with shape torch.Size([2560, 2560]) from checkpoint,
        the shape in current model is torch.Size([2544, 2560])
    size mismatch for layers.0.mlp.w1.weight:
        copying a param with shape torch.Size([10240, 2560]) from checkpoint,
        the shape in current model is torch.Size([12288, 2560])
    ... (160 total mismatches across all 20 layers)
```

---

## Root Cause

The checkpoint at `checkpoints/havoc_phase1_sft/checkpoint_step_50000` was trained with specific model architecture parameters:

**Checkpoint Configuration:**
- `d_model`: 2560
- `num_layers`: 20
- `num_heads`: 32
- `num_kv_heads`: 4
- `mlp.hidden_dim`: 10240
- `mlp.activation`: "gelu"

**Default Configuration (in code):**
- `d_model`: 3072
- `num_layers`: 22
- `num_heads`: 24 (from `AttentionConfig`)
- `num_kv_heads`: 4 (from `AttentionConfig`)
- `mlp.hidden_dim`: 12288 (from `MLPConfig`)
- `mlp.activation`: "swiglu" (from `MLPConfig`)

The original `chat_havoc.py` script only loaded the top-level config parameters (`vocab_size`, `d_model`, `num_layers`, `max_seq_len`, special token IDs) but **did not load the nested `attention` and `mlp` configurations** from the checkpoint's `config.json`.

This caused the model to be instantiated with default `AttentionConfig` and `MLPConfig` values, which didn't match the checkpoint's actual architecture.

---

## Solution

Modified `chat_havoc.py` to properly reconstruct the **complete** model configuration from the checkpoint's `config.json`, including nested configurations:

### Changes Made

1. **Import nested config classes:**
   ```python
   from havoc_core.attention import AttentionConfig
   from havoc_core.mlp import MLPConfig
   ```

2. **Build AttentionConfig from checkpoint:**
   ```python
   attn_cfg = AttentionConfig(
       num_heads=cfg_raw["attention"]["num_heads"],
       head_dim=cfg_raw["attention"]["head_dim"],
       num_kv_heads=cfg_raw["attention"]["num_kv_heads"],
       dropout=cfg_raw["attention"]["dropout"],
       rotary_dim=cfg_raw["attention"]["rotary_dim"],
       rope_theta=cfg_raw["attention"]["rope_theta"],
       bias=cfg_raw["attention"]["bias"],
   )
   ```

3. **Build MLPConfig from checkpoint:**
   ```python
   mlp_cfg = MLPConfig(
       hidden_dim=cfg_raw["mlp"]["hidden_dim"],
       activation=cfg_raw["mlp"]["activation"],
       dropout=cfg_raw["mlp"]["dropout"],
   )
   ```

4. **Pass nested configs to HavocConfig:**
   ```python
   model_cfg = HavocConfig(
       vocab_size=cfg_raw["vocab_size"],
       d_model=cfg_raw["d_model"],
       num_layers=cfg_raw["num_layers"],
       max_seq_len=cfg_raw["max_seq_len"],
       attention=attn_cfg,           # ← Now included
       mlp=mlp_cfg,                  # ← Now included
       dropout=cfg_raw["dropout"],
       layer_norm_eps=cfg_raw["layer_norm_eps"],
       initializer_range=cfg_raw["initializer_range"],
       pad_token_id=cfg_raw["pad_token_id"],
       bos_token_id=cfg_raw["bos_token_id"],
       eos_token_id=cfg_raw["eos_token_id"],
   )
   ```

5. **Added parameter count display:**
   ```python
   n_params = sum(p.numel() for p in model.parameters())
   print(f"Model parameters: {n_params/1e9:.2f}B")
   ```

---

## Verification

After the fix, the script successfully loads the checkpoint:

```bash
$ python chat_havoc.py

=== HAVOC CHAT INTERFACE LOADED ===

Model parameters: 2.05B
Model loaded successfully!

You:
```

**Test script created:** `test_chat.py` successfully loads the model and runs inference (verified with 3 test prompts).

---

## Model Architecture (Checkpoint)

The successfully loaded model has the following architecture:

- **Parameters:** ~2.05B
- **Layers:** 20
- **d_model:** 2560
- **Attention:**
  - 32 query heads
  - 4 key-value heads (Grouped Query Attention)
  - head_dim: 80 (computed as d_model / num_heads = 2560 / 32)
  - RoPE theta: 10000.0
  - No bias
- **MLP:**
  - hidden_dim: 10240 (4x d_model)
  - Activation: GELU
- **Vocabulary:** 70,000 tokens
- **Max sequence length:** 1024 tokens

---

## Key Takeaways

1. **Always load complete checkpoint configuration:** When loading a checkpoint, ensure ALL nested configuration objects are reconstructed from the saved `config.json`, not just top-level parameters.

2. **Configuration dataclasses:** The HAVOC codebase uses a hierarchy of dataclasses:
   - `HavocConfig` (top-level model config)
     - `AttentionConfig` (attention mechanism params)
     - `MLPConfig` (feed-forward network params)

3. **Default values can cause mismatches:** Default values in dataclasses are useful for creating new models, but when loading checkpoints, these defaults should be overridden with the checkpoint's actual configuration.

4. **Checkpoint compatibility:** This fix ensures that `chat_havoc.py` can load any HAVOC checkpoint regardless of its architecture, as long as the `config.json` is complete.

---

## Files Modified

- ✅ [`/workspace/SLM/chat_havoc.py`](chat_havoc.py) - Fixed config loading

## Files Created

- ✅ [`/workspace/SLM/test_chat.py`](test_chat.py) - Test script for verification
- ✅ [`/workspace/SLM/CHAT_FIX_SUMMARY.md`](CHAT_FIX_SUMMARY.md) - This summary

---

## Testing

To test the fixed script:

```bash
# Interactive chat
python chat_havoc.py

# Automated test
python test_chat.py
```

Both scripts now successfully load the checkpoint and can perform inference.

---

**Status:** Issue resolved. The chat interface is now functional and can load checkpoints with arbitrary architectures as long as the configuration is properly saved in `config.json`.
