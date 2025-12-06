cat << 'EOF' > scripts/phase1_pretrain_7b.py
"""
PHASE 1: Pretrain HAVOC-7B from Scratch

Train base language model on 100B tokens (domain-specific + general knowledge).

Usage (Single GPU):
    python scripts/phase1_pretrain_7b.py --data-dir data --checkpoint-dir checkpoints/phase1

Usage (Multi-GPU with torchrun):
    torchrun --nproc_per_node=2 scripts/phase1_pretrain_7b.py \
        --data-dir /workspace/data \
        --tokenizer-path /workspace/SLM/artifacts/tokenizer \
        --checkpoint-dir /workspace/SLM/checkpoints/phase1_h200 \
        --batch-size 4 \
        --gradient-accumulation-steps 8 \
        --max-seq-len 2048
"""

import argparse
import os
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import sentencepiece as spm

from havoc_core.config_7b import Havoc7BConfig, OptimizedTrainingConfig
from havoc_core.model.prime_model import HavocPrimeModel
from havoc_training.optimized_trainer import OptimizedTrainer


class StreamingTextDataset(Dataset):
    """Memory-mapped streaming dataset - no pre-tokenization needed!"""

    def __init__(self, file_paths: list, tokenizer, max_seq_len: int = 2048, samples_per_epoch: int = 10000):
        import random

        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_epoch = samples_per_epoch

        # Build file size index for weighted sampling
        print(f"Indexing {len(file_paths)} files...")
        self.file_sizes = []
        total_size = 0
        for fp in file_paths:
            try:
                size = fp.stat().st_size
                self.file_sizes.append(size)
                total_size += size
            except Exception as e:
                print(f"Skipping {fp}: {e}")

        # Normalize to probabilities
        self.file_probs = [s / total_size for s in self.file_sizes]

        print(f"Dataset initialized with {len(file_paths)} files")
        print(f"Total data size: {total_size / 1e9:.2f} GB")
        print(f"Samples per epoch: {samples_per_epoch:,}")
        print(f"Max sequence length: {max_seq_len}\n")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        import random

        # Weighted random file selection
        file_path = random.choices(self.file_paths, weights=self.file_probs, k=1)[0]

        try:
            # Memory-map or read file with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()

                max_chars = self.max_seq_len * 6
                if file_size > max_chars:
                    start_pos = random.randint(0, file_size - max_chars)
                    f.seek(start_pos)
                    chunk_text = f.read(max_chars)
                else:
                    f.seek(0)
                    chunk_text = f.read()

            # Tokenize
            tokens = self.tokenizer.encode(chunk_text)

            # Pad or truncate
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [0] * (self.max_seq_len - len(tokens))
            else:
                tokens = tokens[:self.max_seq_len]

            return {
                "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
                "labels": torch.tensor(tokens[1:], dtype=torch.long)
            }
        except Exception as e:
            # Fallback for read errors
            print(f"Error reading {file_path}: {e}")
            return {
                "input_ids": torch.zeros(self.max_seq_len - 1, dtype=torch.long),
                "labels": torch.zeros(self.max_seq_len - 1, dtype=torch.long)
            }


def load_tokenizer(tokenizer_path: str):
    """Load SentencePiece tokenizer"""
    sp = spm.SentencePieceProcessor()
    model_file = Path(tokenizer_path) / "tokenizer.model"

    if not model_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {model_file}")

    sp.load(str(model_file))
    return sp


def create_dataloader(data_dir: str, tokenizer, batch_size: int, max_seq_len: int, split: str = "train", use_distributed: bool = False):
    """Create dataloader"""
    data_path = Path(data_dir)
    file_paths = []

    # Check domains
    for domain in ["math", "stats", "engineering", "general", "code"]:
        domain_path = data_path / domain
        if domain_path.exists():
            txt_files = list(domain_path.glob("*.txt"))
            file_paths.extend(txt_files)

    if not file_paths:
        raise ValueError(f"No data files found in {data_dir}")

    # Log only on main process
    is_main = (not use_distributed) or (int(os.environ.get("LOCAL_RANK", 0)) == 0)
    if is_main:
        print(f"Found {len(file_paths)} files for {split}")

    samples_per_epoch = 10000 if split == "train" else 1000
    dataset = StreamingTextDataset(file_paths, tokenizer, max_seq_len, samples_per_epoch=samples_per_epoch)

    sampler = None
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == "train"), drop_last=True)
        shuffle = False
    else:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )


def main():
    parser = argparse.ArgumentParser(description="HAVOC-7B Phase 1")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-every-n-steps", type=int, default=500)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # --- DISTRIBUTED INIT ---
    use_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])

        # --- THE FIX FOR TORCHRUN MASKING ---
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            # torchrun masking active: we only see our own GPU as device 0
            torch.cuda.set_device(0)
        else:
            # No masking: we map rank to device ID
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend="nccl")
        is_main = (local_rank == 0)
    else:
        is_main = True

    # Seed
    torch.manual_seed(42)

    if is_main:
        print("="*60)
        print("PHASE 1: PRETRAINING HAVOC-7B (RELAUNCH)")
        print("="*60)

    tokenizer = load_tokenizer(args.tokenizer_path)
    if is_main:
        print(f"Tokenizer loaded. Vocab: {tokenizer.vocab_size()}")

    # Config & Model
    model_config = Havoc7BConfig()
    model = HavocPrimeModel(model_config, tokenizer=tokenizer)

    if is_main:
        print(f"Model Parameters: {model.get_num_params_billions():.2f}B")

    train_config = OptimizedTrainingConfig(
        model_config=model_config,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=args.save_every_n_steps,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        gradient_checkpointing=True,
        use_flash_attention=True,
        use_amp=True,
        amp_dtype="bfloat16"
    )

    # Loaders
    train_dataloader = create_dataloader(args.data_dir, tokenizer, args.batch_size, args.max_seq_len, "train", use_distributed)
    val_dataloader = create_dataloader(args.data_dir, tokenizer, args.batch_size, args.max_seq_len, "val", use_distributed)

    # Trainer
    trainer = OptimizedTrainer(
        model=model,
        train_config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if is_main:
        print("Starting training...")

    trainer.train()

if __name__ == "__main__":
    main()
EOF
