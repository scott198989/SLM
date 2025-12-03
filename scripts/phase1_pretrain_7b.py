"""
PHASE 1: Pretrain HAVOC-7B from Scratch

Train base language model on 100B tokens (domain-specific + general knowledge).

Usage:
    python scripts/phase1_pretrain_7b.py --data-dir data --checkpoint-dir checkpoints/phase1

Expected training time: ~1000 GPU-hours on RTX 5090 (~42 days)
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from havoc_core.config_7b import Havoc7BConfig, OptimizedTrainingConfig
from havoc_core.model.prime_model import HavocPrimeModel
from havoc_training.optimized_trainer import OptimizedTrainer


class TextDataset(Dataset):
    """Simple text dataset for pretraining"""

    def __init__(self, file_paths: list, tokenizer, max_seq_len: int = 2048):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load all data (for small datasets)
        # For large datasets, use streaming
        self.samples = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split into chunks
                tokens = tokenizer.encode(text)
                for i in range(0, len(tokens), max_seq_len):
                    chunk = tokens[i:i + max_seq_len]
                    if len(chunk) == max_seq_len:  # Only full sequences
                        self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long)
        }


def load_tokenizer(tokenizer_path: str):
    """Load SentencePiece tokenizer"""
    sp = spm.SentencePieceProcessor()
    model_file = Path(tokenizer_path) / "tokenizer.model"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {model_file}. "
            f"Run Phase 0 first: python scripts/phase0_train_tokenizer.py"
        )

    sp.load(str(model_file))
    print(f"Loaded tokenizer: vocab_size={sp.vocab_size()}")
    return sp


def create_dataloader(data_dir: str, tokenizer, batch_size: int, max_seq_len: int, split: str = "train"):
    """Create dataloader for pretraining"""

    data_path = Path(data_dir)

    # Collect files
    file_paths = []
    for domain in ["math", "stats", "engineering", "general", "code"]:
        domain_path = data_path / domain
        if domain_path.exists():
            files = list(domain_path.glob("**/*.txt"))
            file_paths.extend(files)
            print(f"Found {len(files)} files in {domain}/")

    if not file_paths:
        raise ValueError(f"No data files found in {data_dir}")

    print(f"\nTotal files: {len(file_paths)}")

    # Create dataset
    dataset = TextDataset(file_paths, tokenizer, max_seq_len)
    print(f"Total samples: {len(dataset):,}\n")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(description="HAVOC-7B Phase 1: Pretraining")

    # Data
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer", help="Tokenizer path")

    # Model
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase1", help="Checkpoint directory")

    # Training
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep at 1 for 24GB GPU)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32, help="Gradient accumulation")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)

    # Load tokenizer
    print("=" * 70)
    print("PHASE 1: PRETRAINING HAVOC-7B")
    print("=" * 70)
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Create model config
    print("\nCreating model...")
    model_config = Havoc7BConfig()
    model = HavocPrimeModel(model_config, tokenizer=tokenizer)

    print(f"Model parameters: {model.get_num_params_billions():.2f}B")

    # Create training config
    train_config = OptimizedTrainingConfig(
        model_config=model_config,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        gradient_checkpointing=True,  # CRITICAL for 24GB
        use_flash_attention=True,     # CRITICAL for memory
        use_amp=True,
        amp_dtype="bfloat16"
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dataloader = create_dataloader(
        args.data_dir, tokenizer, args.batch_size, args.max_seq_len, split="train"
    )

    val_dataloader = create_dataloader(
        args.data_dir, tokenizer, args.batch_size, args.max_seq_len, split="val"
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = OptimizedTrainer(
        model=model,
        train_config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...\n")
    trainer.train()

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Final checkpoint: {args.checkpoint_dir}/checkpoint_step_{trainer.global_step}")
    print("\nNext step: Run Phase 2 (SFT)")
    print(f"  python scripts/phase2_sft_7b.py --checkpoint {args.checkpoint_dir}/checkpoint_step_{trainer.global_step}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
