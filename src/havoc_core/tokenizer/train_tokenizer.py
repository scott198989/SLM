from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm
import yaml

from havoc_core.config import TokenizerTrainingConfig
from havoc_core.tokenizer.vocab_utils import (
    TokenizerMetadata,
    get_char_normalization_map,
    register_special_tokens,
    sample_domain_strings,
)


def normalize_text(line: str, apply_char_normalization: bool = True) -> str:
    """
    Normalize text with special handling for math/engineering content.

    Args:
        line: Input text line
        apply_char_normalization: Whether to apply character-level normalization

    Returns:
        Normalized text
    """
    # Strip leading/trailing whitespace
    text = line.strip()

    # Apply character-level normalization if enabled
    if apply_char_normalization:
        char_map = get_char_normalization_map()
        for old_char, new_char in char_map.items():
            text = text.replace(old_char, new_char)

    # Collapse multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)

    # Normalize multiple hyphens/dashes to single hyphen
    text = re.sub(r"-+", "-", text)

    return text


def iter_corpus(
    paths: List[str],
    normalize: bool = True,
    apply_char_normalization: bool = True,
) -> Iterable[str]:
    """
    Iterate over corpus files, yielding normalized lines.

    Args:
        paths: List of file or directory paths
        normalize: Whether to apply normalization
        apply_char_normalization: Whether to apply character-level normalization

    Yields:
        Text lines from corpus
    """
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for file in p.rglob("*.txt"):
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if normalize:
                            normalized = normalize_text(line, apply_char_normalization)
                            if normalized:  # Skip empty lines
                                yield normalized
                        else:
                            stripped = line.strip()
                            if stripped:
                                yield stripped
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if normalize:
                        normalized = normalize_text(line, apply_char_normalization)
                        if normalized:
                            yield normalized
                    else:
                        stripped = line.strip()
                        if stripped:
                            yield stripped


def train_tokenizer(config: TokenizerTrainingConfig, verbose: bool = True) -> TokenizerMetadata:
    """
    Train a SentencePiece tokenizer with domain-specific vocabulary.

    Args:
        config: Tokenizer training configuration
        verbose: Whether to print progress messages

    Returns:
        TokenizerMetadata with vocabulary information
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Register all special tokens
    special_tokens = register_special_tokens(config.special_tokens)
    # Do not pass core IDs as user-defined symbols; SentencePiece reserves them via *_id.
    core_reserved = {"<unk>", "<pad>", "<bos>", "<eos>", "<s>", "</s>"}
    special_tokens = [tok for tok in special_tokens if tok not in core_reserved]

    if verbose:
        print(f"Training tokenizer with {len(special_tokens)} special tokens")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Model type: {config.model_type}")

    # Build training corpus
    if verbose:
        print("Loading training corpus...")

    training_corpus = []

    # Add domain sample strings first (ensures they're in the corpus)
    domain_samples = sample_domain_strings()
    training_corpus.extend(domain_samples)

    if verbose:
        print(f"Added {len(domain_samples)} domain sample strings")

    # Add corpus from input files
    if config.input_files:
        for line in iter_corpus(
            config.input_files,
            normalize=config.normalize_text,
            apply_char_normalization=True,
        ):
            training_corpus.append(line)

        if verbose:
            print(f"Loaded {len(training_corpus)} total lines from corpus")
    else:
        if verbose:
            print("Warning: No input files specified, using domain samples only")

    # Train SentencePiece model
    model_prefix = Path(config.output_dir) / "tokenizer"

    if verbose:
        print(f"Training SentencePiece model: {model_prefix}.*")

    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(training_corpus),
        model_prefix=str(model_prefix),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        max_sentence_length=config.max_sentence_length,
        input_sentence_size=len(training_corpus),
        shuffle_input_sentence=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=special_tokens,
        hard_vocab_limit=False,  # allow smaller effective vocab when corpus is small
        # Additional settings for better domain coverage
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        add_dummy_prefix=True,  # Helps with whitespace handling
        remove_extra_whitespaces=True,
        normalization_rule_name="identity",  # Don't normalize, we handle it ourselves
    )

    # Save metadata
    metadata = TokenizerMetadata(
        vocab_size=config.vocab_size,
        special_tokens=special_tokens,
        domain_tokens=[tok for tok in special_tokens if tok not in config.special_tokens],
    )

    metadata_path = Path(config.output_dir) / "tokenizer_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.as_dict(), f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nTokenizer training complete!")
        print(f"Model saved to: {model_prefix}.model")
        print(f"Vocab saved to: {model_prefix}.vocab")
        print(f"Metadata saved to: {metadata_path}")
        print(f"\nSpecial tokens: {len(special_tokens)}")
        print(f"Domain tokens: {len(metadata.domain_tokens)}")

    return metadata


def load_config_from_yaml(config_path: str) -> TokenizerTrainingConfig:
    """Load tokenizer configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return TokenizerTrainingConfig(
        vocab_size=config_dict.get("vocab_size", 75000),
        model_type=config_dict.get("model_type", "bpe"),
        special_tokens=config_dict.get("special_tokens", []),
        input_files=config_dict.get("input_files", []),
        output_dir=config_dict.get("output_dir", "artifacts/tokenizer"),
        character_coverage=config_dict.get("character_coverage", 0.9995),
        max_sentence_length=config_dict.get("max_sentence_length", 2048),
        normalize_text=config_dict.get("normalize_text", True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAVOC tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
        help="Input corpus files or directories (overrides config)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bpe", "unigram", "char", "word"],
        default=None,
        help="SentencePiece model type (overrides config)",
    )

    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config:
        print(f"Loading config from: {args.config}")
        cfg = load_config_from_yaml(args.config)
    else:
        print("Using default configuration")
        cfg = TokenizerTrainingConfig()

    # Apply command-line overrides
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.input_files is not None:
        cfg.input_files = args.input_files
    if args.model_type is not None:
        cfg.model_type = args.model_type

    # Train tokenizer
    print("=" * 60)
    print("HAVOC Tokenizer Training")
    print("=" * 60)
    metadata = train_tokenizer(cfg, verbose=True)
    print("=" * 60)
