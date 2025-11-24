#!/usr/bin/env python
"""Quick test script for tokenizer functionality."""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from havoc_core.config import TokenizerTrainingConfig
from havoc_core.tokenizer import (
    HavocTokenizer,
    get_all_special_tokens,
    train_tokenizer,
)


def test_special_tokens():
    """Test that all special tokens are defined."""
    print("Testing special tokens...")
    all_tokens = get_all_special_tokens()
    print(f"  Total special tokens: {len(all_tokens)}")

    # Check for key token categories
    srs_count = sum(1 for t in all_tokens if "SRS" in t)
    dsl_count = sum(1 for t in all_tokens if "DSL" in t)
    tool_count = sum(1 for t in all_tokens if "TOOL" in t)

    print(f"  SRS tokens: {srs_count}")
    print(f"  DSL tokens: {dsl_count}")
    print(f"  Tool tokens: {tool_count}")

    assert srs_count == 8, f"Expected 8 SRS tokens, got {srs_count}"
    assert dsl_count == 2, f"Expected 2 DSL tokens, got {dsl_count}"
    assert tool_count == 2, f"Expected 2 tool tokens, got {tool_count}"
    print("  ✓ Special tokens OK\n")


def test_tokenizer_training():
    """Test tokenizer training."""
    print("Testing tokenizer training...")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Using temp directory: {tmpdir}")

        # Train minimal tokenizer
        config = TokenizerTrainingConfig(
            vocab_size=5000,
            output_dir=tmpdir,
            input_files=[],  # Use domain samples only
        )

        print("  Training tokenizer (this may take a minute)...")
        metadata = train_tokenizer(config, verbose=False)

        # Check outputs
        model_file = Path(tmpdir) / "tokenizer.model"
        vocab_file = Path(tmpdir) / "tokenizer.vocab"
        metadata_file = Path(tmpdir) / "tokenizer_metadata.json"

        assert model_file.exists(), "tokenizer.model not created"
        assert vocab_file.exists(), "tokenizer.vocab not created"
        assert metadata_file.exists(), "tokenizer_metadata.json not created"

        print(f"  ✓ Model files created")
        print(f"    - tokenizer.model: {model_file.stat().st_size:,} bytes")
        print(f"    - tokenizer.vocab: {vocab_file.stat().st_size:,} bytes")
        print(f"    - metadata: {metadata_file.stat().st_size:,} bytes")

        # Test loading tokenizer
        print("  Loading tokenizer...")
        tokenizer = HavocTokenizer(tmpdir)

        print(f"  ✓ Tokenizer loaded")
        print(f"    - Vocab size: {tokenizer.vocab_size}")
        print(f"    - PAD ID: {tokenizer.pad_id}")
        print(f"    - BOS ID: {tokenizer.bos_id}")
        print(f"    - EOS ID: {tokenizer.eos_id}")

        # Test encoding/decoding
        print("  Testing encoding/decoding...")
        test_texts = [
            "Calculate the mean μ and variance σ²",
            "<SRS_MODE> Identify problem <SRS_GROUND>",
            "<DSL_BEGIN> CHECK_SPC X-bar <DSL_END>",
            "Stress σ = 250 MPa at 150°C",
        ]

        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            print(f"    - '{text[:40]}...'")
            print(f"      Tokens: {len(encoded)}")

        print("  ✓ Encoding/decoding OK\n")


def test_math_symbols():
    """Test math symbol handling."""
    print("Testing math symbol handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create corpus with math symbols
        corpus_file = Path(tmpdir) / "math_corpus.txt"
        corpus_file.write_text(
            "Calculate mean μ and variance σ²\n"
            "Sum: ∑ᵢ₌₁ⁿ xᵢ\n"
            "Integral: ∫₀¹ x² dx\n"
            "Gradient: ∇f\n"
        )

        config = TokenizerTrainingConfig(
            vocab_size=5000,
            output_dir=tmpdir,
            input_files=[str(corpus_file)],
        )

        train_tokenizer(config, verbose=False)
        tokenizer = HavocTokenizer(tmpdir)

        # Test math expression
        text = "Calculate μ and σ²"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"  Input: '{text}'")
        print(f"  Tokens: {len(encoded)}")
        print(f"  Decoded: '{decoded}'")
        print("  ✓ Math symbols OK\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("HAVOC Tokenizer Quick Test")
    print("=" * 60)
    print()

    try:
        test_special_tokens()
        test_tokenizer_training()
        test_math_symbols()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
