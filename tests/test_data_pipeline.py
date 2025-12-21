"""Tests for the HAVOC-7B data pipeline.

Tests cover:
- DataSource file discovery and iteration
- Preprocessing (symbol normalization, DSL extraction, reasoning annotation)
- CausalLMDataset mixture weighting
- Sample packing
- BOS/EOS handling
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from havoc_core.config import DataMixtureConfig
from havoc_data.dataset import CausalLMDataset, MixturePolicy
from havoc_data.preprocess import (
    annotate_reasoning_traces,
    extract_dsl_blocks,
    is_malformed,
    normalize_symbols,
    normalize_text,
)
from havoc_data.sources import DataSource


class DummyTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self):
        self.vocab = {}
        self.next_id = 0

    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            tokens.append(self.vocab[word])
        return tokens

    def decode(self, tokens: list) -> str:
        """Decode token IDs to text."""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse_vocab.get(t, "<unk>") for t in tokens)


# ============================================================================
# Preprocessing Tests
# ============================================================================


def test_normalize_symbols():
    """Test Unicode symbol normalization."""
    text = "Temperature: 25°C, σ = 1.5, μ = 10.0"
    normalized = normalize_symbols(text)
    assert "deg" in normalized
    assert "sigma" in normalized
    assert "mu" in normalized
    assert "°" not in normalized
    assert "σ" not in normalized
    assert "μ" not in normalized


def test_extract_dsl_blocks():
    """Test DSL block extraction."""
    text = """
    Some text before

    ```dsl
    EXPERIMENT test
      FACTORS a, b
    END
    ```

    Some text after
    """

    modified, blocks = extract_dsl_blocks(text)

    assert len(blocks) == 1
    assert "EXPERIMENT test" in blocks[0]
    assert "<DSL_BEGIN>" in modified
    assert "<DSL_END>" in modified
    assert "```dsl" not in modified


def test_annotate_reasoning_traces():
    """Test reasoning trace annotation."""
    text = """
    Problem: Calculate 2+2

    <think>
    This is simple addition.
    2 + 2 = 4
    </think>

    Answer: 4
    """

    annotated = annotate_reasoning_traces(text)

    assert "<REASONING_BEGIN>" in annotated
    assert "<REASONING_END>" in annotated
    assert "<think>" not in annotated
    assert "</think>" not in annotated
    assert "This is simple addition" in annotated


def test_is_malformed():
    """Test malformed line detection."""
    # Valid lines
    assert not is_malformed("This is a normal line of text.")
    assert not is_malformed("x = 2 + 2, y = 3.5")

    # Too short
    assert is_malformed("short")

    # Too many special characters
    assert is_malformed("@#$%^&*()_+@#$%^&*()_+@#$%^&*()_+")

    # No alphanumeric content
    assert is_malformed("... --- ... --- ...")


def test_normalize_text_integration():
    """Test full text normalization pipeline."""
    text = """
    Temperature: 150°C, σ = 2.5

    ```dsl
    DESIGN factorial
    END
    ```

    <think>
    Calculate mean
    </think>
    """

    normalized = normalize_text(text, extract_dsl=True, annotate_reasoning=True)

    assert "deg" in normalized  # Symbol normalization
    assert "sigma" in normalized
    assert "<DSL_BEGIN>" in normalized  # DSL extraction
    assert "<REASONING_BEGIN>" in normalized  # Reasoning annotation


# ============================================================================
# DataSource Tests
# ============================================================================


def test_datasource_txt_files(tmp_path):
    """Test DataSource with text files."""
    # Create test files
    (tmp_path / "file1.txt").write_text("Content 1")
    (tmp_path / "file2.txt").write_text("Content 2")

    source = DataSource(name="test", paths=[str(tmp_path)], weight=1.0)

    files = source.files()
    assert len(files) == 2


def test_datasource_jsonl_files(tmp_path):
    """Test DataSource with JSONL files."""
    # Create test JSONL file
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"text": "Line 1", "meta": "A"}) + "\n")
        f.write(json.dumps({"text": "Line 2", "meta": "B"}) + "\n")

    source = DataSource(
        name="test",
        paths=[str(tmp_path)],
        weight=1.0,
        file_type="jsonl",
        text_field="text",
        metadata_fields=["meta"],
    )

    docs = list(source.iter_documents())
    assert len(docs) == 2
    assert docs[0]["text"] == "Line 1"
    assert docs[0]["metadata"]["meta"] == "A"
    assert docs[1]["text"] == "Line 2"


def test_datasource_domain_tagging(tmp_path):
    """Test domain tagging in DataSource."""
    (tmp_path / "math.txt").write_text("x^2 + y^2 = z^2")

    source = DataSource(
        name="mathematics", paths=[str(tmp_path)], weight=2.0, domain="math"
    )

    docs = list(source.iter_documents())
    assert docs[0]["domain"] == "math"
    assert docs[0]["source"] == "mathematics"


# ============================================================================
# MixturePolicy Tests
# ============================================================================


def test_mixture_policy_weights():
    """Test mixture policy weight normalization."""
    sources = [
        DataSource("s1", [], weight=1.0),
        DataSource("s2", [], weight=2.0),
        DataSource("s3", [], weight=3.0),
    ]

    policy = MixturePolicy(sources)

    # Check normalized weights
    assert len(policy.normalized_weights) == 3
    assert sum(policy.normalized_weights) == pytest.approx(1.0)
    assert policy.normalized_weights[0] == pytest.approx(1.0 / 6.0)
    assert policy.normalized_weights[1] == pytest.approx(2.0 / 6.0)
    assert policy.normalized_weights[2] == pytest.approx(3.0 / 6.0)


def test_mixture_policy_sampling():
    """Test mixture policy sampling distribution."""
    sources = [
        DataSource("high_weight", [], weight=10.0),
        DataSource("low_weight", [], weight=1.0),
    ]

    policy = MixturePolicy(sources)

    # Sample many times and check distribution
    samples = [policy.sample().name for _ in range(1000)]
    high_count = samples.count("high_weight")
    low_count = samples.count("low_weight")

    # Should be roughly 10:1 ratio
    ratio = high_count / low_count
    assert 7.0 < ratio < 13.0  # Allow some variance


# ============================================================================
# CausalLMDataset Tests
# ============================================================================


def test_dataset_basic_iteration(tmp_path):
    """Test basic dataset iteration."""
    # Create test data
    (tmp_path / "data.txt").write_text("This is a test document.")

    source = DataSource(name="test", paths=[str(tmp_path)], weight=1.0)
    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=[source],
        mixture=mixture,
        max_seq_len=128,
        enable_packing=False,
    )

    # Get one example
    iterator = iter(dataset)
    example = next(iterator)

    assert isinstance(example.input_ids, torch.Tensor)
    assert isinstance(example.attention_mask, torch.Tensor)
    assert isinstance(example.labels, torch.Tensor)
    assert example.input_ids.shape == (128,)
    assert example.attention_mask.shape == (128,)
    assert example.labels.shape == (128,)


def test_dataset_bos_eos_handling(tmp_path):
    """Test BOS/EOS token handling."""
    (tmp_path / "data.txt").write_text("Hello world")

    source = DataSource(name="test", paths=[str(tmp_path)], weight=1.0)
    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=[source],
        mixture=mixture,
        max_seq_len=128,
        bos_token_id=1,
        eos_token_id=2,
        enable_packing=False,
    )

    iterator = iter(dataset)
    example = next(iterator)

    # First token should be BOS
    assert example.input_ids[0].item() == 1

    # Find first non-padding token from the end
    # (should be EOS)
    non_padding = example.attention_mask.nonzero(as_tuple=True)[0]
    if len(non_padding) > 0:
        last_real_token_idx = non_padding[-1].item()
        assert example.input_ids[last_real_token_idx].item() == 2


def test_dataset_padding_and_masking(tmp_path):
    """Test padding and attention masking."""
    (tmp_path / "data.txt").write_text("Short text")

    source = DataSource(name="test", paths=[str(tmp_path)], weight=1.0)
    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=[source],
        mixture=mixture,
        max_seq_len=128,
        pad_token_id=0,
        enable_packing=False,
    )

    iterator = iter(dataset)
    example = next(iterator)

    # Check that padding tokens have attention_mask = 0
    for i in range(128):
        if example.input_ids[i].item() == 0:
            assert example.attention_mask[i].item() == 0
            assert example.labels[i].item() == -100  # Masked label


def test_dataset_mixture_weighting(tmp_path):
    """Test that mixture weights affect sampling."""
    # Create two sources with different weights
    dir1 = tmp_path / "high"
    dir2 = tmp_path / "low"
    dir1.mkdir()
    dir2.mkdir()

    (dir1 / "data.txt").write_text("High weight source")
    (dir2 / "data.txt").write_text("Low weight source")

    sources = [
        DataSource(name="high", paths=[str(dir1)], weight=10.0, domain="high"),
        DataSource(name="low", paths=[str(dir2)], weight=1.0, domain="low"),
    ]

    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer, sources=sources, mixture=mixture, max_seq_len=128
    )

    # Sample many examples and track which source they came from
    # This is approximate since we can't directly check the source
    # But we can verify the mixture policy
    policy = dataset.mixture_policy
    stats = policy.get_mixture_stats()

    assert stats["high"] > stats["low"]
    assert abs(stats["high"] - 10.0 / 11.0) < 0.01
    assert abs(stats["low"] - 1.0 / 11.0) < 0.01


def test_dataset_sample_packing(tmp_path):
    """Test sample packing feature."""
    # Create short documents
    (tmp_path / "doc1.txt").write_text("Short doc one")
    (tmp_path / "doc2.txt").write_text("Short doc two")

    source = DataSource(name="test", paths=[str(tmp_path)], weight=1.0)
    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=[source],
        mixture=mixture,
        max_seq_len=128,
        enable_packing=True,
    )

    iterator = iter(dataset)
    example = next(iterator)

    # With packing, we should have multiple documents in one sequence
    # Count EOS tokens (token id 2)
    eos_count = (example.input_ids == 2).sum().item()

    # Should have at least 1 EOS (more if multiple docs packed)
    assert eos_count >= 1


def test_dataset_preprocessing_integration(tmp_path):
    """Test that preprocessing is applied during dataset iteration."""
    text_with_symbols = "Temperature σ = 5.0°C"
    (tmp_path / "data.txt").write_text(text_with_symbols)

    source = DataSource(name="test", paths=[str(tmp_path)], weight=1.0)
    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=[source],
        mixture=mixture,
        max_seq_len=128,
        extract_dsl=True,
        annotate_reasoning=True,
    )

    # Just verify it doesn't crash
    iterator = iter(dataset)
    example = next(iterator)

    assert example is not None


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_pipeline_integration(tmp_path):
    """Test complete pipeline from files to dataset."""
    # Create multi-domain data
    math_dir = tmp_path / "math"
    stats_dir = tmp_path / "stats"
    math_dir.mkdir()
    stats_dir.mkdir()

    (math_dir / "calculus.txt").write_text("Derivative of x^2 is 2x")
    (stats_dir / "ttest.txt").write_text("T-test compares means")

    sources = [
        DataSource(name="math", paths=[str(math_dir)], weight=2.0, domain="mathematics"),
        DataSource(
            name="stats", paths=[str(stats_dir)], weight=1.0, domain="statistics"
        ),
    ]

    mixture = DataMixtureConfig(max_sequence_length=128)
    tokenizer = DummyTokenizer()

    dataset = CausalLMDataset(
        tokenizer=tokenizer, sources=sources, mixture=mixture, max_seq_len=128
    )

    # Sample multiple examples
    iterator = iter(dataset)
    examples = [next(iterator) for _ in range(5)]

    assert len(examples) == 5
    for ex in examples:
        assert ex.input_ids.shape == (128,)
        assert ex.attention_mask.sum() > 0  # Has some real tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
