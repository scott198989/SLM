#!/usr/bin/env python3
"""
HAVOC Tokenizer Training Script (70k vocab)

CANONICAL TOKENIZER CONFIGURATION:
- vocab_size: 70000
- model_type: BPE
- pad_id: 0, bos_id: 1, eos_id: 2, unk_id: 3

RUNPOD ENVIRONMENT:
- Hardware: AMD MI300X (ROCm)
- Repo root: /workspace/SLM
- Dataset root: /workspace/data
- Pretrain corpus: /workspace/data/pretrain
- Tokenizer output: /workspace/SLM/artifacts/tokenizer

This script trains a SentencePiece BPE tokenizer that matches the model's
embedding dimensions. Run this BEFORE training to ensure tokenizer/model compatibility.

Usage (on RunPod MI300X):
    python scripts/train_tokenizer_70k.py \
        --corpus /workspace/data/pretrain \
        --output /workspace/SLM/artifacts/tokenizer

The corpus can be:
- A directory containing .txt files
- A single .txt file
- Multiple paths (space-separated)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sentencepiece as spm

# =============================================================================
# CANONICAL CONFIGURATION - DO NOT CHANGE WITHOUT UPDATING MODEL CONFIGS
# =============================================================================
CANONICAL_VOCAB_SIZE = 70000
CANONICAL_PAD_ID = 0
CANONICAL_BOS_ID = 1
CANONICAL_EOS_ID = 2
CANONICAL_UNK_ID = 3

# Domain-specific special tokens for HAVOC
SPECIAL_TOKENS = [
    # SRS reasoning tokens
    "<SRS_MODE>",
    "<SRS_GROUND>",
    "<SRS_PLAN>",
    "<SRS_EXECUTE>",
    "<SRS_ARGUE>",
    "<SRS_ARBITER>",
    "<SRS_AUDIT>",
    "<SRS_ANSWER>",
    # Tool tokens
    "<tool>",
    "</tool>",
    "<ref>",
    "</ref>",
    "<plan>",
    "</plan>",
    "<exec>",
    "</exec>",
    "<argue>",
    "</argue>",
    "<arbiter>",
    "</arbiter>",
    "<audit>",
    "</audit>",
    # Reasoning tokens
    "<reason>",
    "</reason>",
    "<advocate>",
    "</advocate>",
    "<attack>",
    "</attack>",
    "<pragmatist>",
    "</pragmatist>",
    # DSL tokens
    "<DSL_BEGIN>",
    "<DSL_END>",
    "<TOOL_MATH>",
    "<TOOL_STATS>",
    # Engineering symbols
    "<ENG_SYMBOL_START>",
    "<ENG_SYMBOL_END>",
]

# Math/engineering symbols to add as user-defined symbols
MATH_SYMBOLS = [
    "∑", "∏", "∫", "∂", "∇", "√", "≈", "≠", "≤", "≥", "±", "×", "÷", "∞",
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
    "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ",
    "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
]

# Engineering units and domain terms
DOMAIN_TERMS = [
    "MPa", "GPa", "kPa", "N/m", "kg/m³", "m/s", "m/s²",
    "kW", "MW", "kWh", "°C", "°F", "Hz", "kHz", "MHz", "GHz",
    "ANOVA", "p-value", "control_chart", "Box-Behnken", "Taguchi",
    "Cpk", "Cp", "Ppk", "X-bar", "R-chart", "UCL", "LCL", "CL",
    "factorial", "fractional_factorial", "central_composite", "Plackett-Burman",
]


def iter_corpus(paths: list[str]):
    """Iterate over corpus files, yielding lines."""
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for file in sorted(p.rglob("*.txt")):
                print(f"  Reading: {file}")
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
        elif p.exists():
            print(f"  Reading: {p}")
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        else:
            print(f"  WARNING: Path not found: {p}")


def train_tokenizer(
    corpus_paths: list[str],
    output_dir: str,
    vocab_size: int = CANONICAL_VOCAB_SIZE,
):
    """Train SentencePiece BPE tokenizer with canonical settings."""

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = Path(output_dir) / "tokenizer"

    print("=" * 70)
    print("HAVOC TOKENIZER TRAINING")
    print("=" * 70)
    print(f"Vocab size:  {vocab_size}")
    print(f"Model type:  BPE")
    print(f"Output dir:  {output_dir}")
    print(f"Token IDs:   pad={CANONICAL_PAD_ID}, bos={CANONICAL_BOS_ID}, "
          f"eos={CANONICAL_EOS_ID}, unk={CANONICAL_UNK_ID}")
    print("=" * 70)

    # Collect all user-defined symbols (excluding reserved core tokens)
    user_symbols = SPECIAL_TOKENS + MATH_SYMBOLS + DOMAIN_TERMS

    print(f"\nUser-defined symbols: {len(user_symbols)}")
    print(f"Loading corpus from: {corpus_paths}")

    # Build training corpus in memory
    print("\nLoading corpus...")
    corpus = list(iter_corpus(corpus_paths))
    print(f"Loaded {len(corpus):,} lines")

    if len(corpus) == 0:
        print("ERROR: No corpus data found!")
        print("Please provide a corpus directory or file with .txt files.")
        sys.exit(1)

    # Train SentencePiece
    print(f"\nTraining SentencePiece model...")
    print(f"  Model prefix: {model_prefix}")

    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        max_sentence_length=4096,
        input_sentence_size=len(corpus),
        shuffle_input_sentence=True,
        # CANONICAL TOKEN IDS
        pad_id=CANONICAL_PAD_ID,
        bos_id=CANONICAL_BOS_ID,
        eos_id=CANONICAL_EOS_ID,
        unk_id=CANONICAL_UNK_ID,
        # User-defined symbols
        user_defined_symbols=user_symbols,
        # Allow smaller vocab if corpus is small
        hard_vocab_limit=False,
        # Text processing
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="identity",
    )

    # Verify the trained tokenizer
    print("\nVerifying tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
    actual_vocab_size = sp.get_piece_size()
    print(f"  Actual vocab size: {actual_vocab_size}")
    print(f"  pad_id: {sp.pad_id()}")
    print(f"  bos_id: {sp.bos_id()}")
    print(f"  eos_id: {sp.eos_id()}")
    print(f"  unk_id: {sp.unk_id()}")

    # Save metadata
    metadata = {
        "vocab_size": actual_vocab_size,
        "target_vocab_size": vocab_size,
        "pad_id": CANONICAL_PAD_ID,
        "bos_id": CANONICAL_BOS_ID,
        "eos_id": CANONICAL_EOS_ID,
        "unk_id": CANONICAL_UNK_ID,
        "special_tokens": SPECIAL_TOKENS,
        "domain_tokens": MATH_SYMBOLS + DOMAIN_TERMS,
        "model_type": "bpe",
    }

    metadata_path = Path(output_dir) / "tokenizer_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Create HuggingFace-compatible config for PreTrainedTokenizerFast
    hf_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 4096,
        "sp_model_kwargs": {},
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    config_path = Path(output_dir) / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    print("\n" + "=" * 70)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model:     {model_prefix}.model")
    print(f"Vocab:     {model_prefix}.vocab")
    print(f"Metadata:  {metadata_path}")
    print(f"HF Config: {config_path}")
    print("=" * 70)

    # Warn if vocab size mismatch
    if actual_vocab_size != vocab_size:
        print(f"\n⚠️  WARNING: Actual vocab size ({actual_vocab_size}) differs from target ({vocab_size})")
        print("   This may happen with small corpora. Update model config vocab_size to match!")
        print(f"   Set vocab_size={actual_vocab_size} in HavocConfig and Havoc7BConfig")

    return actual_vocab_size


def main():
    parser = argparse.ArgumentParser(
        description="Train HAVOC tokenizer (70k vocab)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from a corpus directory
  python scripts/train_tokenizer_70k.py --corpus /workspace/data/pretrain

  # Train from multiple sources
  python scripts/train_tokenizer_70k.py --corpus /data/math /data/general /data/eng

  # Custom output directory
  python scripts/train_tokenizer_70k.py --corpus /data --output /workspace/tokenizer
        """,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to corpus directory or file(s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/SLM/artifacts/tokenizer",
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=CANONICAL_VOCAB_SIZE,
        help=f"Target vocabulary size (default: {CANONICAL_VOCAB_SIZE})",
    )

    args = parser.parse_args()

    actual_size = train_tokenizer(
        corpus_paths=args.corpus,
        output_dir=args.output,
        vocab_size=args.vocab_size,
    )

    print(f"\n✅ Tokenizer ready at: {args.output}")
    print(f"   Vocab size: {actual_size}")


if __name__ == "__main__":
    main()
