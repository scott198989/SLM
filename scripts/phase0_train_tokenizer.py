"""
PHASE 0: Train SentencePiece Tokenizer

Train a 70k vocabulary tokenizer with reasoning tokens for HAVOC-7B.

Usage:
    python scripts/phase0_train_tokenizer.py --input-dir data/corpus --output-dir artifacts/tokenizer
"""

import argparse
from pathlib import Path
import sentencepiece as spm


def train_tokenizer(
    input_files: list,
    output_dir: str,
    vocab_size: int = 70000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995
):
    """
    Train SentencePiece tokenizer

    Args:
        input_files: List of input text files
        output_dir: Output directory for tokenizer
        vocab_size: Vocabulary size
        model_type: Model type (bpe or unigram)
        character_coverage: Character coverage
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_prefix = str(output_path / "tokenizer")

    # Special tokens for HAVOC-7B
    user_defined_symbols = [
        "<pad>",       # 0
        "<bos>",       # 1
        "<eos>",       # 2
        "<unk>",       # 3
        "<mask>",      # 4
        # Reasoning tokens
        "<reason>",    # 10
        "</reason>",   # 11
        "<tool>",      # 12
        "</tool>",     # 13
        "<advocate>",  # 14
        "</advocate>", # 15
        "<attack>",    # 16
        "</attack>",   # 17
        "<pragmatist>",# 18
        "</pragmatist>",# 19
    ]

    # Train tokenizer
    print("=" * 70)
    print("TRAINING SENTENCEPIECE TOKENIZER")
    print("=" * 70)
    print(f"Input files: {len(input_files)}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Model type: {model_type}")
    print(f"Character coverage: {character_coverage}")
    print(f"Special tokens: {len(user_defined_symbols)}")
    print("=" * 70 + "\n")

    spm.SentencePieceTrainer.train(
        input=",".join(str(f) for f in input_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        user_defined_symbols=user_defined_symbols,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        max_sentence_length=4096,
        num_threads=8,
        train_extremely_large_corpus=True,
        input_sentence_size=10000000,  # Sample 10M sentences
        shuffle_input_sentence=True
    )

    print("\n" + "=" * 70)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocab saved to: {model_prefix}.vocab")
    print("=" * 70 + "\n")

    # Test tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    test_sentences = [
        "Design a Box-Behnken DOE for temperature, pressure, and speed.",
        "<reason>This is a test of reasoning tokens.</reason>",
        "<tool>{\"tool\": \"python_math\", \"args\": {}}</tool>"
    ]

    print("Testing tokenizer:\n")
    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        print(f"Input: {sentence}")
        print(f"Tokens: {tokens[:20]}...")
        print(f"Vocab size: {sp.vocab_size()}\n")


def main():
    parser = argparse.ArgumentParser(description="Train HAVOC-7B tokenizer (Phase 0)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing training corpus")
    parser.add_argument("--output-dir", type=str, default="artifacts/tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=70000, help="Vocabulary size")
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram"], help="Model type")

    args = parser.parse_args()

    # Collect input files
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob("**/*.txt"))

    if not input_files:
        print(f"Error: No .txt files found in {input_dir}")
        return

    # Train tokenizer
    train_tokenizer(
        input_files=input_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type
    )


if __name__ == "__main__":
    main()
