"""
PHASE 1: Pretrain HAVOC-7B from Scratch

Train base language model on 100B tokens (domain-specific + general knowledge).

Usage:
    python scripts/phase1_pretrain_7b.py --data-dir data --checkpoint-dir checkpoints/phase1

Expected training time:
    - H200 (141GB): ~18 hours (recommended: batch_size=16, grad_accum=4, seq_len=4096)
    - RTX 5090 (24GB): ~1000 GPU-hours (~42 days)
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
    """Lazy-loading text dataset for pretraining with LRU cache"""

    def __init__(self, file_paths: list, tokenizer, max_seq_len: int = 2048, samples_per_epoch: int = 10000):
        import random
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_epoch = samples_per_epoch
        self.cache = {}  # Simple cache for recently accessed files
        self.max_cache_size = 20  # Cache up to 20 files

        print(f"Dataset initialized with {len(file_paths)} files")
        print(f"Samples per epoch: {samples_per_epoch:,}")
        print(f"Max sequence length: {max_seq_len}\n")

    def __len__(self):
        return self.samples_per_epoch

    def _load_and_tokenize(self, file_path):
        """Load and tokenize file with caching"""
        file_str = str(file_path)

        # Check cache
        if file_str in self.cache:
            return self.cache[file_str]

        # Read and tokenize
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = self.tokenizer.encode(text)

        # Add to cache (LRU eviction)
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))

        self.cache[file_str] = tokens
        return tokens

    def __getitem__(self, idx):
        import random

        # Pick a random file
        file_path = random.choice(self.file_paths)

        # Load and tokenize (with caching)
        tokens = self._load_and_tokenize(file_path)

        # Sample a random chunk
        if len(tokens) < self.max_seq_len:
            # Pad short sequences
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        elif len(tokens) > self.max_seq_len:
            # Take random chunk from long sequences
            max_start = len(tokens) - self.max_seq_len
            start = random.randint(0, max_start)
            tokens = tokens[start:start + self.max_seq_len]

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
            files = list(domain_path.glob("*.txt"))
            file_paths.extend(files)
            print(f"Found {len(files)} files in {domain}/")

    if not file_paths:
        raise ValueError(f"No data files found in {data_dir}")

    print(f"\nTotal files: {len(file_paths)}")

    # Create dataset
    # Use 10k samples per epoch for faster iteration with large datasets
    samples_per_epoch = 10000 if split == "train" else 1000
    dataset = TextDataset(file_paths, tokenizer, max_seq_len, samples_per_epoch=samples_per_epoch)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with lazy loading
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (1 for 24GB GPU, 16-32 for H200)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32, help="Gradient accumulation (32 for 24GB, 4 for H200)")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length (2048 for 24GB, 4096 for H200)")

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
        gradient_checkpointing=False,  # Disabled for H200 (141GB VRAM) - prioritize speed
        use_flash_attention=True,      # Keep enabled for Hopper architecture
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
