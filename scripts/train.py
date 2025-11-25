#!/usr/bin/env python3
"""
HAVOC-7B Training Script

Usage:
    python scripts/train.py --config configs/training/default_training.yaml
    python scripts/train.py --config configs/training/default_training.yaml --resume checkpoints/checkpoint_step_1000

This script is the main entrypoint for training the HAVOC-7B model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from havoc_core.config import HavocConfig, TrainingConfig, DataMixtureConfig
from havoc_core.model.transformer import HavocModel
from havoc_data.dataset import CausalLMDataset
from havoc_data.sources import DataSource, TextFileSource
from havoc_training.trainer import Trainer


def load_config_from_yaml(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Parse model config
    if "model" in config_dict:
        model_config_dict = config_dict["model"]
        from dataclasses import fields
        from havoc_core.config import AttentionConfig, MLPConfig

        # Parse nested attention config
        if "attention" in model_config_dict:
            attn_dict = model_config_dict["attention"]
            attention_config = AttentionConfig(**attn_dict)
            model_config_dict["attention"] = attention_config

        # Parse nested MLP config
        if "mlp" in model_config_dict:
            mlp_dict = model_config_dict["mlp"]
            mlp_config = MLPConfig(**mlp_dict)
            model_config_dict["mlp"] = mlp_config

        model_config = HavocConfig(**model_config_dict)
        config_dict["model_config"] = model_config
        del config_dict["model"]

    # Parse data config
    if "data" in config_dict:
        data_config = DataMixtureConfig(**config_dict["data"])
        config_dict["data_config"] = data_config
        del config_dict["data"]

    return TrainingConfig(**config_dict)


def create_dummy_tokenizer(vocab_size: int):
    """
    Create a simple dummy tokenizer for testing.
    In production, replace with actual SentencePiece tokenizer.
    """
    class DummyTokenizer:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2

        def __call__(self, text: str) -> list[int]:
            # Simple character-level tokenization for demo
            # In production, use trained SentencePiece tokenizer
            tokens = [self.bos_token_id]
            for char in text[:100]:  # Truncate for demo
                tokens.append(min(ord(char) % self.vocab_size, self.vocab_size - 1))
            tokens.append(self.eos_token_id)
            return tokens

    return DummyTokenizer(vocab_size)


def create_datasets(config: TrainingConfig):
    """
    Create training and validation datasets.

    NOTE: This is a placeholder. In production:
    1. Load actual text corpora (math, stats, engineering, general)
    2. Use trained SentencePiece tokenizer
    3. Apply proper data mixture ratios
    """
    # Check if data sources are provided
    data_dir = Path("data")
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        print("\n" + "=" * 80)
        print("WARNING: No data directory found or no .txt files in data/")
        print("=" * 80)
        print("\nTo train with real data:")
        print("  1. Create a 'data/' directory")
        print("  2. Add your text files (*.txt)")
        print("  3. Organize by domain: data/math/, data/stats/, data/general/, etc.")
        print("\nFor now, creating a minimal dummy dataset for testing...")
        print("=" * 80 + "\n")

        # Create dummy dataset for testing
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size: int, seq_len: int):
                self.size = size
                self.seq_len = seq_len

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # Random token IDs for testing
                input_ids = torch.randint(0, 1000, (self.seq_len,))
                attention_mask = torch.ones_like(input_ids)
                return input_ids, attention_mask

        train_dataset = DummyDataset(size=1000, seq_len=config.data_config.max_sequence_length)
        val_dataset = DummyDataset(size=100, seq_len=config.data_config.max_sequence_length)

        return train_dataset, val_dataset

    # Real data loading (when data files exist)
    print(f"Loading data from {data_dir}...")

    # Create tokenizer
    tokenizer = create_dummy_tokenizer(config.model_config.vocab_size)

    # Create data sources
    # TODO: Implement proper data source weights based on config
    sources = []
    for txt_file in data_dir.rglob("*.txt"):
        source = TextFileSource(name=txt_file.stem, paths=[str(txt_file)], weight=1.0)
        sources.append(source)

    if not sources:
        raise ValueError(f"No .txt files found in {data_dir}")

    print(f"Found {len(sources)} data sources")

    # Split sources into train/val (90/10 split)
    split_idx = int(len(sources) * 0.9)
    train_sources = sources[:split_idx] if split_idx > 0 else sources
    val_sources = sources[split_idx:] if split_idx < len(sources) else sources[:1]

    # Create datasets
    train_dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=train_sources,
        mixture=config.data_config,
    )

    val_dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=val_sources,
        mixture=config.data_config,
    )

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train HAVOC-7B model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/default_training.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to load (alias for --resume).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu). Overrides config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Overrides config.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate. Overrides config.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps. Overrides config.",
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config_from_yaml(args.config)

    # Override config with CLI arguments
    if args.resume or args.checkpoint:
        config.resume_from_checkpoint = args.resume or args.checkpoint
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps

    # Print configuration
    print("\n" + "=" * 80)
    print("HAVOC-7B Training Configuration")
    print("=" * 80)
    print(f"Model: {config.model_config.num_layers} layers, {config.model_config.d_model} d_model")
    print(f"Vocab size: {config.model_config.vocab_size}")
    print(f"Max sequence length: {config.model_config.max_seq_len}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Max grad norm: {config.max_grad_norm}")
    print(f"LR scheduler: {config.lr_scheduler_type}")
    print(f"Mixed precision: {config.use_amp} ({config.amp_dtype if config.use_amp else 'N/A'})")
    print(f"Device: {config.device}")
    print(f"Max epochs: {config.max_epochs}")
    if config.max_steps:
        print(f"Max steps: {config.max_steps}")
    print(f"Log eval examples: {config.log_eval_examples}")
    print("=" * 80 + "\n")

    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}\n")

    # Initialize model
    print("Initializing model...")
    model = HavocModel(config.model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e9:.2f}B\n")

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("checkpoint_interrupted")
        print("Checkpoint saved. You can resume with: --resume checkpoints/checkpoint_interrupted")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save checkpoint...")
        try:
            trainer.save_checkpoint("checkpoint_error")
            print("Checkpoint saved to: checkpoints/checkpoint_error")
        except:
            print("Failed to save checkpoint.")
        raise

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
