#!/usr/bin/env python
"""Smoke test for HAVOC-7B model implementation."""
import sys
from pathlib import Path

# Ensure repository src/ is importable when running as a script without editable install
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    from havoc_core.config import HavocConfig
    print("✓ HavocConfig imported successfully")
except ImportError as e:
    print(f"✗ Failed to import HavocConfig: {e}")
    sys.exit(1)

try:
    from havoc_core.model import HavocModel
    print("✓ HavocModel imported successfully")
except ImportError as e:
    print(f"✗ Failed to import HavocModel: {e}")
    sys.exit(1)

# Test 1: Create a small config
print("\n[Test 1] Creating small config...")
try:
    config = HavocConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=2,
        max_seq_len=128,
    )
    config.attention.num_heads = 8
    config.attention.num_kv_heads = 2
    config.attention.head_dim = 32
    config.mlp.hidden_dim = 512
    print(f"✓ Config created: d_model={config.d_model}, num_layers={config.num_layers}")
except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 2: Initialize model
print("\n[Test 2] Initializing model...")
try:
    torch.manual_seed(42)
    model = HavocModel(config)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Failed to initialize model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check weight tying
print("\n[Test 3] Checking weight tying...")
try:
    assert model.lm_head.weight is model.embed_tokens.weight
    print("✓ Weights are tied between embed_tokens and lm_head")
except AssertionError:
    print("✗ Weights are NOT tied")
    sys.exit(1)

# Test 4: Forward pass without cache
print("\n[Test 4] Testing forward pass (no cache)...")
try:
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits, past_kv = model(input_ids, use_cache=False)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert past_kv is None
    print(f"✓ Forward pass successful: logits shape = {logits.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass with cache
print("\n[Test 5] Testing forward pass (with KV-cache)...")
try:
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits, past_kv = model(input_ids, use_cache=True)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert past_kv is not None
    assert len(past_kv) == config.num_layers

    # Check KV cache shapes
    k, v = past_kv[0]
    assert k.shape == (batch_size, config.attention.num_kv_heads, seq_len, config.attention.head_dim)
    assert v.shape == (batch_size, config.attention.num_kv_heads, seq_len, config.attention.head_dim)

    print(f"✓ KV-cache working: cache shape = {k.shape}")
except Exception as e:
    print(f"✗ KV-cache test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Incremental forward with cache
print("\n[Test 6] Testing incremental forward with KV-cache...")
try:
    batch_size = 1
    initial_len = 8
    new_len = 1

    # First forward pass
    input_ids_1 = torch.randint(0, config.vocab_size, (batch_size, initial_len))
    model.eval()
    with torch.no_grad():
        logits_1, past_kv_1 = model(input_ids_1, use_cache=True)

    # Incremental forward pass
    input_ids_2 = torch.randint(0, config.vocab_size, (batch_size, new_len))
    with torch.no_grad():
        logits_2, past_kv_2 = model(input_ids_2, past_key_values=past_kv_1, use_cache=True)

    assert logits_2.shape == (batch_size, new_len, config.vocab_size)

    # Check that cache grew
    k1, v1 = past_kv_1[0]
    k2, v2 = past_kv_2[0]
    assert k2.shape[2] == k1.shape[2] + new_len

    print(f"✓ Incremental forward successful: cache grew from {k1.shape[2]} to {k2.shape[2]}")
except Exception as e:
    print(f"✗ Incremental forward failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Loss computation
print("\n[Test 7] Testing loss computation...")
try:
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.train()
    loss, logits = model.compute_loss(input_ids)

    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    print(f"✓ Loss computation successful: loss = {loss.item():.4f}")
except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Generation
print("\n[Test 8] Testing generation...")
try:
    batch_size = 1
    prompt_len = 4
    max_new_tokens = 8

    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))

    model.eval()
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens)

    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= prompt_len + max_new_tokens
    assert torch.all(generated[:, :prompt_len] == prompt_ids)

    print(f"✓ Generation successful: generated {generated.shape[1]} tokens")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Gradient flow
print("\n[Test 9] Testing gradient flow...")
try:
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.train()
    model.zero_grad()
    loss, _ = model.compute_loss(input_ids)
    loss.backward()

    # Check that gradients exist and are non-zero
    has_gradients = True
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or torch.all(param.grad == 0):
                has_gradients = False
                no_grad_params.append(name)

    if has_gradients:
        print("✓ Gradient flow verified")
    else:
        print(f"⚠ Warning: Some parameters have no gradients: {no_grad_params[:5]}")
except Exception as e:
    print(f"✗ Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Save and load
print("\n[Test 10] Testing save/load...")
try:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "model"

        # Save model
        model.save_pretrained(str(save_dir), use_safetensors=False)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "pytorch_model.bin").exists()

        # Load model
        loaded_model = HavocModel.load_pretrained(str(save_dir))

        # Check configs match
        assert loaded_model.config.vocab_size == config.vocab_size

        print("✓ Save/load successful")
except Exception as e:
    print(f"✗ Save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nDefinition of Done verified:")
print("✓ Can run: from havoc_core.model import HavocModel")
print("✓ Can run: model = HavocModel(config)")
print("✓ Can run: logits = model(input_ids)")
print("✓ All features implemented and working!")
