"""
PHASE 2: Supervised Fine-Tuning (SFT) for HAVOC-7B

Teach the model to:
1. Use reasoning tokens (<reason>...</reason>)
2. Call tools (<tool>...</tool>)
3. Apply PRIME adversarial reasoning (<advocate>, <attack>, <pragmatist>)
4. Solve domain-specific problems

Usage:
    python scripts/phase2_sft_7b.py \\
        --checkpoint checkpoints/phase1/checkpoint_step_100000 \\
        --sft-data data/sft \\
        --output-dir checkpoints/phase2

Expected training time: ~100 GPU-hours on RTX 5090 (~4 days)
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from havoc_core.config_7b import Havoc7BConfig, OptimizedTrainingConfig
from havoc_core.model.prime_model import HavocPrimeModel
from havoc_training.optimized_trainer import OptimizedTrainer


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with reasoning examples"""

    def __init__(self, jsonl_files: list, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        # Load JSONL files
        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    self.examples.append(example)

        print(f"Loaded {len(self.examples)} SFT examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format: {"prompt": "...", "completion": "..."}
        prompt = example["prompt"]
        completion = example["completion"]

        # Concatenate prompt + completion
        full_text = f"<bos>{prompt}\n{completion}<eos>"

        # Tokenize
        tokens = self.tokenizer.encode(full_text)

        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        # Pad if needed
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        # Create labels (mask prompt, only train on completion)
        prompt_tokens = self.tokenizer.encode(f"<bos>{prompt}\n")
        labels = [-100] * len(prompt_tokens) + tokens[len(prompt_tokens):]

        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long)
        }


def load_sft_data(data_dir: str, tokenizer, batch_size: int, max_seq_len: int):
    """Load SFT data from JSONL files"""

    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("**/*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {data_dir}")

    print(f"\nFound {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    # Create dataset
    dataset = SFTDataset(jsonl_files, tokenizer, max_seq_len)

    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="HAVOC-7B Phase 2: Supervised Fine-Tuning")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Phase 1 checkpoint path")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer", help="Tokenizer path")

    # Data
    parser.add_argument("--sft-data", type=str, required=True, help="SFT data directory (JSONL files)")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase2", help="Output checkpoint directory")

    # Training
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2: SUPERVISED FINE-TUNING (SFT)")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(args.tokenizer_path) / "tokenizer.model"))

    # Load pretrained model
    print(f"\nLoading pretrained model from {args.checkpoint}...")
    model = HavocPrimeModel.from_pretrained(args.checkpoint, device="cuda")
    print(f"Model parameters: {model.get_num_params_billions():.2f}B")

    # Create training config (lower LR for fine-tuning)
    train_config = OptimizedTrainingConfig(
        model_config=model.config,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,  # Lower LR
        min_learning_rate=args.learning_rate / 10,
        max_steps=args.max_steps,
        checkpoint_dir=args.output_dir,
        warmup_steps=500,  # Shorter warmup
        gradient_checkpointing=True,
        use_flash_attention=True,
        use_amp=True,
        amp_dtype="bfloat16"
    )

    # Load SFT data
    print("\nLoading SFT data...")
    train_dataloader, val_dataloader = load_sft_data(
        args.sft_data, sp, args.batch_size, args.max_seq_len
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = OptimizedTrainer(
        model=model,
        train_config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=sp
    )

    # Train
    print("\nStarting SFT training...\n")
    trainer.train()

    print("\n" + "=" * 70)
    print("PHASE 2 (SFT) COMPLETE")
    print("=" * 70)
    print(f"Final checkpoint: {args.output_dir}/checkpoint_step_{trainer.global_step}")
    print("\nNext step: Run Phase 3 (Conversational Polish)")
    print(f"  python scripts/phase3_polish_7b.py --checkpoint {args.output_dir}/checkpoint_step_{trainer.global_step}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
