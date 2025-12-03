"""
PHASE 3: Conversational Polish for HAVOC-7B

Polish the model for:
1. Natural conversational flow
2. Helpful, harmless, honest responses
3. Proper tone and formatting
4. Safety and alignment

Usage:
    python scripts/phase3_polish_7b.py \\
        --checkpoint checkpoints/phase2/checkpoint_step_10000 \\
        --polish-data data/polish \\
        --output-dir checkpoints/phase3

Expected training time: ~50 GPU-hours on RTX 5090 (~2 days)
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


class ConversationalDataset(Dataset):
    """Dataset for conversational polish training"""

    def __init__(self, jsonl_files: list, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.conversations = []

        # Load conversations
        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conversation = json.loads(line)
                    self.conversations.append(conversation)

        print(f"Loaded {len(self.conversations)} conversations")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        messages = conversation["messages"]

        # Build conversation text
        text_parts = ["<bos>"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"{role.capitalize()}: {content}")

        text_parts.append("<eos>")
        full_text = "\n\n".join(text_parts)

        # Tokenize
        tokens = self.tokenizer.encode(full_text)

        # Truncate/pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        elif len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        # Create labels (train on assistant responses only)
        labels = []
        current_pos = 0
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            msg_text = f"{role.capitalize()}: {content}"
            msg_tokens = self.tokenizer.encode(msg_text)

            if role == "assistant":
                # Train on assistant response
                labels.extend(msg_tokens)
            else:
                # Mask user input
                labels.extend([-100] * len(msg_tokens))

            current_pos += len(msg_tokens)

        # Pad labels
        if len(labels) < len(tokens):
            labels = labels + [-100] * (len(tokens) - len(labels))

        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:self.max_seq_len], dtype=torch.long)
        }


def load_polish_data(data_dir: str, tokenizer, batch_size: int, max_seq_len: int):
    """Load conversational polish data"""

    data_path = Path(data_dir)
    jsonl_files = list(data_path.glob("**/*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {data_dir}")

    print(f"\nFound {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    # Create dataset
    dataset = ConversationalDataset(jsonl_files, tokenizer, max_seq_len)

    # Split 90/10
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
    parser = argparse.ArgumentParser(description="HAVOC-7B Phase 3: Conversational Polish")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Phase 2 checkpoint path")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer", help="Tokenizer path")

    # Data
    parser.add_argument("--polish-data", type=str, required=True, help="Polish data directory (JSONL)")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase3", help="Output directory")

    # Training
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate (very low for polish)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3: CONVERSATIONAL POLISH")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(args.tokenizer_path) / "tokenizer.model"))

    # Load SFT model
    print(f"\nLoading model from {args.checkpoint}...")
    model = HavocPrimeModel.from_pretrained(args.checkpoint, device="cuda")
    print(f"Model parameters: {model.get_num_params_billions():.2f}B")

    # Create training config (very low LR)
    train_config = OptimizedTrainingConfig(
        model_config=model.config,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,  # Very low LR
        min_learning_rate=args.learning_rate / 10,
        max_steps=args.max_steps,
        checkpoint_dir=args.output_dir,
        warmup_steps=200,  # Short warmup
        gradient_checkpointing=True,
        use_flash_attention=True,
        use_amp=True,
        amp_dtype="bfloat16"
    )

    # Load polish data
    print("\nLoading polish data...")
    train_dataloader, val_dataloader = load_polish_data(
        args.polish_data, sp, args.batch_size, args.max_seq_len
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
    print("\nStarting polish training...\n")
    trainer.train()

    print("\n" + "=" * 70)
    print("PHASE 3 (POLISH) COMPLETE")
    print("=" * 70)
    print(f"Final checkpoint: {args.output_dir}/checkpoint_step_{trainer.global_step}")
    print("\n✅ HAVOC-7B TRAINING COMPLETE!")
    print("\nYour model is ready to use:")
    print(f"  model = HavocPrimeModel.from_pretrained('{args.output_dir}/checkpoint_step_{trainer.global_step}')")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
