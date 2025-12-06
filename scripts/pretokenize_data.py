"""
Pre-tokenize dataset for fast training

This script reads all chunk_*.txt files and creates corresponding
chunk_*.pt files containing pre-tokenized data.

Usage:
    python scripts/pretokenize_data.py --data-dir /workspace/data/general --tokenizer-path /workspace/SLM/artifacts/tokenizer
"""

import argparse
import torch
from pathlib import Path
import sentencepiece as spm
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing chunk_*.txt files")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as data-dir)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(args.tokenizer_path) / "tokenizer.model"))
    print(f"Loaded tokenizer: vocab_size={sp.vocab_size()}\n")

    # Find all text files
    txt_files = sorted(data_dir.glob("chunk_*.txt"))
    print(f"Found {len(txt_files)} files to tokenize\n")

    # Process each file
    for txt_file in tqdm(txt_files, desc="Tokenizing files"):
        pt_file = output_dir / f"{txt_file.stem}.pt"

        # Skip if already tokenized
        if pt_file.exists():
            print(f"  [SKIP] {pt_file.name} already exists")
            continue

        # Read and tokenize
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = sp.encode(text)

        # Save as tensor
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        torch.save(token_tensor, pt_file)

        print(f"  [DONE] {txt_file.name} -> {pt_file.name} ({len(tokens):,} tokens)")

    print("\n" + "=" * 70)
    print("Pre-tokenization complete!")
    print("=" * 70)
    print(f"Tokenized files saved to: {output_dir}")
    print("\nNow update your training script to use .pt files instead of .txt files")


if __name__ == "__main__":
    main()
