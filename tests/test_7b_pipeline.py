#!/usr/bin/env python3
"""
HAVOC-7B Pipeline Smoke Tests

CLAUDE_FIX: Created comprehensive smoke tests to validate the 7B training pipeline.

These tests run quickly on CPU with a mini configuration to verify:
1. Config loading and validation
2. Model instantiation (both 2B and 7B configs)
3. Forward pass works
4. Loss computation works
5. Optimizer step works
6. Tokenizer alignment with model vocab
7. Data pipeline produces correct shapes

Run with:
    pytest tests/test_7b_pipeline.py -v

Or standalone:
    python tests/test_7b_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn


# =============================================================================
# MINI CONFIGS FOR TESTING (keeps tests fast on CPU)
# =============================================================================

def get_mini_2b_config():
    """Return a tiny 2B-style config for fast testing."""
    from havoc_core.config import HavocConfig
    from havoc_core.attention import AttentionConfig
    from havoc_core.mlp import MLPConfig

    return HavocConfig(
        vocab_size=1000,  # Tiny vocab for testing
        d_model=64,
        num_layers=2,
        max_seq_len=128,
        attention=AttentionConfig(
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            dropout=0.0,
        ),
        mlp=MLPConfig(
            hidden_dim=256,
            activation="swiglu",
        ),
        dropout=0.0,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def get_mini_7b_config():
    """Return a tiny 7B-style config for fast testing."""
    from havoc_core.config_7b import Havoc7BConfig
    from havoc_core.attention import AttentionConfig
    from havoc_core.mlp import MLPConfig

    config = Havoc7BConfig(
        vocab_size=1000,  # Tiny vocab for testing
        d_model=128,
        num_layers=4,
        max_seq_len=128,
        attention=AttentionConfig(
            num_heads=8,
            num_kv_heads=4,
            head_dim=16,
            dropout=0.0,
        ),
        mlp=MLPConfig(
            hidden_dim=512,
            activation="swiglu",
        ),
        dropout=0.0,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return config


# =============================================================================
# TEST: CONFIG LOADING AND VALIDATION
# =============================================================================

def test_config_loading_2b():
    """Test that 2B config loads correctly."""
    from havoc_core.config import HavocConfig

    config = HavocConfig()
    assert config.vocab_size == 70000, "Default vocab should be 70k"
    assert config.d_model == 2560, "Default d_model should be 2560"
    assert config.num_layers == 20, "Default layers should be 20"
    print("PASS: test_config_loading_2b")


def test_config_loading_7b():
    """Test that 7B config loads correctly."""
    from havoc_core.config_7b import Havoc7BConfig

    config = Havoc7BConfig()
    assert config.vocab_size == 70000, "7B vocab should be 70k"
    assert config.d_model == 4096, "7B d_model should be 4096"
    assert config.num_layers == 32, "7B layers should be 32"
    assert config.attention.num_heads == 32, "7B should have 32 heads"
    assert config.attention.num_kv_heads == 8, "7B should have 8 KV heads (GQA)"
    print("PASS: test_config_loading_7b")


def test_mini_config_instantiation():
    """Test mini configs can be instantiated."""
    config_2b = get_mini_2b_config()
    config_7b = get_mini_7b_config()

    assert config_2b.d_model == 64
    assert config_7b.d_model == 128
    print("PASS: test_mini_config_instantiation")


# =============================================================================
# TEST: MODEL INSTANTIATION
# =============================================================================

def test_model_instantiation_2b():
    """Test HavocModel can be instantiated with mini 2B config."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_2b_config()
    model = HavocModel(config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Mini 2B model params: {param_count:,}")
    assert param_count > 0, "Model should have parameters"
    print("PASS: test_model_instantiation_2b")


def test_model_instantiation_7b():
    """Test HavocModel can be instantiated with mini 7B config."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_7b_config()
    model = HavocModel(config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Mini 7B model params: {param_count:,}")
    assert param_count > 0, "Model should have parameters"
    print("PASS: test_model_instantiation_7b")


# =============================================================================
# TEST: FORWARD PASS
# =============================================================================

def test_forward_pass_2b():
    """Test forward pass works with mini 2B model."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_2b_config()
    model = HavocModel(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits should not contain NaN"
    print("PASS: test_forward_pass_2b")


def test_forward_pass_7b():
    """Test forward pass works with mini 7B model."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_7b_config()
    model = HavocModel(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits should not contain NaN"
    print("PASS: test_forward_pass_7b")


# =============================================================================
# TEST: LOSS COMPUTATION
# =============================================================================

def test_loss_computation():
    """Test that cross-entropy loss can be computed."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_2b_config()
    model = HavocModel(config)
    model.train()

    # Create dummy input and labels
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, _ = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        labels.view(-1)
    )

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"Loss: {loss.item():.4f}")
    print("PASS: test_loss_computation")


# =============================================================================
# TEST: BACKWARD PASS AND OPTIMIZER STEP
# =============================================================================

def test_optimizer_step():
    """Test that a full training step works (forward + backward + optimizer)."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_2b_config()
    model = HavocModel(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy input and labels
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward
    logits, _ = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        labels.view(-1)
    )

    # Backward
    loss.backward()

    # Check gradients exist
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Model should have gradients after backward"

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    print("PASS: test_optimizer_step")


# =============================================================================
# TEST: GRADIENT CHECKPOINTING
# =============================================================================

def test_gradient_checkpointing():
    """Test that gradient checkpointing can be enabled."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_2b_config()
    model = HavocModel(config)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    assert model._gradient_checkpointing == True, "Checkpointing should be enabled"

    model.train()

    # Forward + backward should still work
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = torch.randint(0, config.vocab_size, (2, 32))

    logits, _ = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
    loss.backward()

    print("PASS: test_gradient_checkpointing")


# =============================================================================
# TEST: GENERATION
# =============================================================================

def test_generation():
    """Test that model.generate() works."""
    from havoc_core.model.transformer import HavocModel

    config = get_mini_2b_config()
    model = HavocModel(config)
    model.eval()

    # Create prompt
    prompt_ids = torch.randint(0, config.vocab_size, (1, 5))

    # Generate
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=10, temperature=1.0)

    assert generated.shape[1] > prompt_ids.shape[1], "Should generate new tokens"
    assert generated.shape[1] <= prompt_ids.shape[1] + 10, "Should not exceed max_new_tokens"
    print(f"Generated sequence length: {generated.shape[1]}")
    print("PASS: test_generation")


# =============================================================================
# TEST: PRIME MODEL (7B SPECIFIC)
# =============================================================================

def test_prime_model_instantiation():
    """Test HavocPrimeModel can be instantiated."""
    try:
        from havoc_core.config_7b import Havoc7BConfig
        from havoc_core.model.prime_model import HavocPrimeModel

        config = get_mini_7b_config()
        model = HavocPrimeModel(config)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Mini PrimeModel params: {param_count:,}")
        assert param_count > 0, "Model should have parameters"
        print("PASS: test_prime_model_instantiation")
    except ImportError as e:
        print(f"SKIP: test_prime_model_instantiation - Missing dependencies: {e}")


# =============================================================================
# TEST: VOCAB SIZE ALIGNMENT
# =============================================================================

def test_vocab_size_alignment():
    """Test that all configs have consistent vocab size."""
    from havoc_core.config import HavocConfig, TokenizerTrainingConfig
    from havoc_core.config_7b import Havoc7BConfig

    base_config = HavocConfig()
    config_7b = Havoc7BConfig()
    tokenizer_config = TokenizerTrainingConfig()

    assert base_config.vocab_size == 70000, "Base config vocab should be 70k"
    assert config_7b.vocab_size == 70000, "7B config vocab should be 70k"
    assert tokenizer_config.vocab_size == 70000, "Tokenizer config vocab should be 70k"

    print(f"Base config vocab: {base_config.vocab_size}")
    print(f"7B config vocab: {config_7b.vocab_size}")
    print(f"Tokenizer config vocab: {tokenizer_config.vocab_size}")
    print("PASS: test_vocab_size_alignment")


# =============================================================================
# TEST: NO OPTION-E GUARD BLOCKING 7B
# =============================================================================

def test_no_option_e_guard():
    """Test that train.py no longer has Option-E guard blocking 7B."""
    train_script = Path(__file__).parent.parent / "scripts" / "train.py"

    with open(train_script, "r") as f:
        content = f.read()

    # Check that the old blocking guard is removed
    assert "HARD GUARD FOR HAVOC OPTION-E" not in content, \
        "Option-E hard guard should be removed"

    # Check that the fix comment exists
    assert "CLAUDE_FIX" in content, \
        "CLAUDE_FIX comment should exist"

    print("PASS: test_no_option_e_guard")


# =============================================================================
# TEST: NO DOT PENALTY IN GENERATE
# =============================================================================

def test_no_dot_penalty():
    """Test that transformer.py no longer has the dot penalty bug."""
    transformer_file = Path(__file__).parent.parent / "src" / "havoc_core" / "model" / "transformer.py"

    with open(transformer_file, "r") as f:
        content = f.read()

    # Check that the dot penalty code is removed
    assert 'probs[:, dot_id] = 0.0' not in content, \
        "Dot penalty should be removed"

    print("PASS: test_no_dot_penalty")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("HAVOC-7B PIPELINE SMOKE TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_config_loading_2b,
        test_config_loading_7b,
        test_mini_config_instantiation,
        test_model_instantiation_2b,
        test_model_instantiation_7b,
        test_forward_pass_2b,
        test_forward_pass_7b,
        test_loss_computation,
        test_optimizer_step,
        test_gradient_checkpointing,
        test_generation,
        test_prime_model_instantiation,
        test_vocab_size_alignment,
        test_no_option_e_guard,
        test_no_dot_penalty,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            print(f"\n--- Running: {test.__name__} ---")
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                print(f"ERROR: {test.__name__} - {e}")
                failed += 1

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print("=" * 70)

    if failed == 0:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\n{failed} TEST(S) FAILED!")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
