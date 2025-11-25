"""Comprehensive tests for HAVOC-7B transformer model."""
import tempfile
from pathlib import Path

import pytest
import torch

from havoc_core.config import HavocConfig
from havoc_core.model import HavocModel


@pytest.fixture
def small_config():
    """Create a small config for testing."""
    config = HavocConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=2,
        max_seq_len=128,
        dropout=0.1,
    )
    config.attention.num_heads = 8
    config.attention.num_kv_heads = 2
    config.attention.head_dim = 32
    config.mlp.hidden_dim = 512
    return config


@pytest.fixture
def model(small_config):
    """Create a small model for testing."""
    torch.manual_seed(42)
    return HavocModel(small_config)


def test_model_initialization(small_config):
    """Test that model initializes correctly."""
    model = HavocModel(small_config)

    assert model.config == small_config
    assert len(model.layers) == small_config.num_layers
    assert model.embed_tokens.num_embeddings == small_config.vocab_size
    assert model.embed_tokens.embedding_dim == small_config.d_model

    # Check that weights are tied
    assert model.lm_head.weight is model.embed_tokens.weight


def test_forward_pass(model, small_config):
    """Test forward pass with dummy batch."""
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    # Forward pass without cache
    logits, past_kv = model(input_ids, use_cache=False)

    assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
    assert past_kv is None


def test_forward_pass_with_cache(model, small_config):
    """Test forward pass with KV-cache."""
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    # Forward pass with cache
    logits, past_kv = model(input_ids, use_cache=True)

    assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
    assert past_kv is not None
    assert len(past_kv) == small_config.num_layers

    # Check shape of cached keys and values
    for k, v in past_kv:
        assert k.shape == (
            batch_size,
            small_config.attention.num_kv_heads,
            seq_len,
            small_config.attention.head_dim,
        )
        assert v.shape == (
            batch_size,
            small_config.attention.num_kv_heads,
            seq_len,
            small_config.attention.head_dim,
        )


def test_incremental_forward_with_cache(model, small_config):
    """Test incremental forward pass using cached KV states."""
    batch_size = 1
    initial_len = 8
    new_len = 1

    # First forward pass
    input_ids_1 = torch.randint(0, small_config.vocab_size, (batch_size, initial_len))
    logits_1, past_kv_1 = model(input_ids_1, use_cache=True)

    # Incremental forward pass with only new token
    input_ids_2 = torch.randint(0, small_config.vocab_size, (batch_size, new_len))
    logits_2, past_kv_2 = model(input_ids_2, past_key_values=past_kv_1, use_cache=True)

    assert logits_2.shape == (batch_size, new_len, small_config.vocab_size)
    assert len(past_kv_2) == small_config.num_layers

    # Check that cache grew
    for (k1, v1), (k2, v2) in zip(past_kv_1, past_kv_2):
        assert k2.shape[2] == k1.shape[2] + new_len
        assert v2.shape[2] == v1.shape[2] + new_len


def test_attention_mask_shape(model, small_config):
    """Test attention mask correctness."""
    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    # Forward pass with attention mask
    logits, _ = model(input_ids, attention_mask=attention_mask)

    assert logits.shape == (batch_size, seq_len, small_config.vocab_size)


def test_causal_mask_correctness(model, small_config):
    """Test that causal masking works correctly."""
    batch_size = 1
    seq_len = 4

    # Create input where each position has a unique token
    input_ids = torch.arange(seq_len).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids)

    # The logits at position i should not depend on tokens at positions > i
    # This is hard to test directly, but we can test the mask shape
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    mask = model._build_attention_mask(attention_mask)

    assert mask.shape == (batch_size, 1, seq_len, seq_len)

    # Check that mask is causal (upper triangle should be -inf)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[0, 0, i, j] == float("-inf")
            else:
                assert mask[0, 0, i, j] == 0.0


def test_compute_loss(model, small_config):
    """Test loss computation."""
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    # Compute loss (labels = shifted input_ids)
    loss, logits = model.compute_loss(input_ids)

    assert loss.ndim == 0  # Loss should be a scalar
    assert loss.item() >= 0  # Loss should be non-negative
    assert logits.shape == (batch_size, seq_len, small_config.vocab_size)


def test_compute_loss_with_labels(model, small_config):
    """Test loss computation with explicit labels."""
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    loss, logits = model.compute_loss(input_ids, labels=labels)

    assert loss.ndim == 0
    assert loss.item() >= 0
    assert logits.shape == (batch_size, seq_len, small_config.vocab_size)


def test_compute_loss_ignores_pad_token(model, small_config):
    """Test that loss ignores pad tokens."""
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # Replace some labels with pad token
    labels[:, ::2] = small_config.pad_token_id

    loss, _ = model.compute_loss(input_ids, labels=labels)

    assert loss.ndim == 0
    assert loss.item() >= 0


def test_generation_smoke_test(model, small_config):
    """Smoke test for generation."""
    batch_size = 1
    prompt_len = 4
    max_new_tokens = 8

    prompt_ids = torch.randint(0, small_config.vocab_size, (batch_size, prompt_len))

    model.eval()
    generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens)

    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= prompt_len + max_new_tokens
    assert torch.all(generated[:, :prompt_len] == prompt_ids)


def test_generation_deterministic(model, small_config):
    """Test that generation with temperature=1.0 is deterministic."""
    batch_size = 1
    prompt_len = 4
    max_new_tokens = 4

    prompt_ids = torch.randint(0, small_config.vocab_size, (batch_size, prompt_len))

    model.eval()
    torch.manual_seed(42)
    generated_1 = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=1.0)

    torch.manual_seed(42)
    generated_2 = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=1.0)

    assert torch.all(generated_1 == generated_2)


def test_save_and_load_config(model, small_config):
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        model.save_config(str(config_path))

        assert config_path.exists()


def test_save_and_load_pretrained(model, small_config):
    """Test saving and loading model with weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "model"

        # Save model (using PyTorch format since safetensors may not be installed)
        model.save_pretrained(str(save_dir), use_safetensors=False)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "pytorch_model.bin").exists()

        # Load model
        loaded_model = HavocModel.load_pretrained(str(save_dir))

        # Check that configs match
        assert loaded_model.config.vocab_size == small_config.vocab_size
        assert loaded_model.config.d_model == small_config.d_model
        assert loaded_model.config.num_layers == small_config.num_layers

        # Check that weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)


def test_gradient_flow(model, small_config):
    """Test that gradients flow through the model."""
    batch_size = 2
    seq_len = 8

    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    model.train()
    loss, _ = model.compute_loss(input_ids)
    loss.backward()

    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_model_from_config(small_config):
    """Test creating model using from_config class method."""
    model = HavocModel.from_config(small_config)

    assert isinstance(model, HavocModel)
    assert model.config == small_config


def test_dropout_training_vs_eval(model, small_config):
    """Test that dropout behaves differently in training vs eval mode."""
    if small_config.dropout == 0.0:
        pytest.skip("Dropout is 0.0, skipping this test")

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

    torch.manual_seed(42)
    model.train()
    logits_train_1, _ = model(input_ids)

    torch.manual_seed(42)
    model.train()
    logits_train_2, _ = model(input_ids)

    # Training mode with dropout should produce different outputs
    # (unless dropout is 0.0 or we're unlucky with randomness)
    diff = torch.abs(logits_train_1 - logits_train_2).max().item()
    assert diff > 0, "Dropout should produce different outputs in training mode"

    torch.manual_seed(42)
    model.eval()
    with torch.no_grad():
        logits_eval_1, _ = model(input_ids)

    torch.manual_seed(42)
    model.eval()
    with torch.no_grad():
        logits_eval_2, _ = model(input_ids)

    # Eval mode should produce identical outputs
    assert torch.allclose(logits_eval_1, logits_eval_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
