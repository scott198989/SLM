from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from havoc_core.config import HavocConfig, TrainingConfig
from havoc_core.model.transformer import HavocModel

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for HAVOC-7B model.

    Handles:
    - Model initialization
    - Optimizer & scheduler setup
    - Mixed precision training (AMP)
    - Gradient accumulation & clipping
    - Checkpoint save/load/resume
    - Training & validation loops
    - Logging
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[HavocModel] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        self._set_seed(config.seed)

        # Model
        if model is None:
            if config.model_config is None:
                raise ValueError("Either model or model_config must be provided")
            self.model = HavocModel(config.model_config)
        else:
            self.model = model
        self.model.to(self.device)

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Optimizer, scheduler, scaler (initialized in setup_training)
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None
        self.scaler: Optional[GradScaler] = None

        # Setup directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_logging(self) -> None:
        """Setup logging to file and console."""
        log_file = Path(self.config.log_dir) / "train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    def setup_training(self) -> None:
        """Initialize optimizer, scheduler, and gradient scaler."""
        # Optimizer: AdamW with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Gradient scaler for mixed precision
        if self.config.use_amp:
            self.scaler = GradScaler()

        logger.info(f"Training setup complete:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        logger.info(f"  Optimizer: AdamW")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Scheduler: {self.config.lr_scheduler_type}")
        logger.info(f"  Mixed precision: {self.config.use_amp} ({self.config.amp_dtype if self.config.use_amp else 'N/A'})")

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup and decay."""
        def lr_lambda(current_step: int) -> float:
            # Warmup
            if current_step < self.config.warmup_steps:
                return current_step / max(1, self.config.warmup_steps)

            # Calculate progress after warmup
            if self.config.max_steps is not None:
                total_steps = self.config.max_steps
            else:
                # Estimate total steps from epochs
                steps_per_epoch = len(self.train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)
                total_steps = steps_per_epoch * self.config.max_epochs

            progress = (current_step - self.config.warmup_steps) / max(1, total_steps - self.config.warmup_steps)
            progress = min(1.0, progress)

            # Apply decay schedule
            if self.config.lr_scheduler_type == "cosine":
                # Cosine annealing
                decay_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_lr_ratio = self.config.min_learning_rate / self.config.learning_rate
                return min_lr_ratio + (1.0 - min_lr_ratio) * decay_ratio
            elif self.config.lr_scheduler_type == "linear":
                # Linear decay
                return max(self.config.min_learning_rate / self.config.learning_rate, 1.0 - progress)
            else:  # constant
                return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> None:
        """Main training loop."""
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided before training")

        # Setup training components
        self.setup_training()

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info(f"  Epochs: {self.config.max_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {len(train_loader) * self.config.max_epochs // self.config.gradient_accumulation_steps}")
        logger.info("=" * 80)

        self.model.train()
        accumulated_loss = 0.0

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            logger.info(f"\n--- Epoch {epoch + 1}/{self.config.max_epochs} ---")

            for step, batch in enumerate(train_loader):
                # Forward pass and loss calculation
                loss = self._training_step(batch, step)
                accumulated_loss += loss

                # Update weights every gradient_accumulation_steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        current_lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step:6d} | "
                            f"Epoch {epoch + 1} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e}"
                        )
                        accumulated_loss = 0.0

                    # Validation
                    if self.val_dataset is not None and self.global_step % self.config.eval_every_n_steps == 0:
                        val_loss, val_perplexity = self.evaluate()
                        logger.info(
                            f"Validation | "
                            f"Step {self.global_step:6d} | "
                            f"Loss: {val_loss:.4f} | "
                            f"Perplexity: {val_perplexity:.2f}"
                        )
                        self.model.train()

                    # Checkpoint saving
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint()

                    # Check if max_steps reached
                    if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                        logger.info(f"Reached max_steps ({self.config.max_steps}). Stopping training.")
                        return

        logger.info("\n" + "=" * 80)
        logger.info("Training complete!")
        logger.info("=" * 80)

    def _training_step(self, batch: tuple, step: int) -> float:
        """Single training step with gradient accumulation."""
        input_ids, attention_mask = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Shift targets for causal LM
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        # Mixed precision forward pass
        if self.config.use_amp:
            dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
            with autocast(dtype=dtype):
                logits, _ = self.model(input_ids, attention_mask=None)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=self.model.config.pad_token_id,
                )
        else:
            logits, _ = self.model(input_ids, attention_mask=None)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.model.config.pad_token_id,
            )

        # Scale loss by gradient accumulation steps
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def _optimizer_step(self) -> None:
        """Optimizer step with gradient clipping."""
        if self.config.use_amp:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        # Optimizer step
        if self.config.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        self.scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad()

    def evaluate(self) -> tuple[float, float]:
        """Evaluate model on validation set."""
        if self.val_dataset is None:
            logger.warning("No validation dataset provided. Skipping evaluation.")
            return 0.0, 0.0

        self.model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= self.config.eval_samples:
                    break

                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # Shift targets
                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()

                # Forward pass
                logits, _ = self.model(input_ids, attention_mask=None)

                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=self.model.config.pad_token_id,
                    reduction="sum",
                )

                # Count non-padding tokens
                num_tokens = (labels != self.model.config.pad_token_id).sum().item()
                total_loss += loss.item()
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        return avg_loss, perplexity

    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> None:
        """Save training checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.global_step}"

        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")

        # Save scheduler state
        torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")

        # Save scaler state
        if self.config.use_amp:
            torch.save(self.scaler.state_dict(), checkpoint_path / "scaler.pt")

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        # Save config
        with open(checkpoint_path / "config.json", "w") as f:
            # Convert dataclass to dict for JSON serialization
            import dataclasses
            config_dict = dataclasses.asdict(self.config)
            json.dump(config_dict, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: int(x.name.split("_")[-1]),
        )

        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                shutil.rmtree(checkpoint)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and resume training."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Load model state
        self.model.load_state_dict(torch.load(checkpoint_path / "model.pt", map_location=self.device))

        # Load optimizer state
        if (checkpoint_path / "optimizer.pt").exists():
            self.optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))

        # Load scheduler state
        if (checkpoint_path / "scheduler.pt").exists():
            self.scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))

        # Load scaler state
        if self.config.use_amp and (checkpoint_path / "scaler.pt").exists():
            self.scaler.load_state_dict(torch.load(checkpoint_path / "scaler.pt"))

        # Load training state
        if (checkpoint_path / "training_state.json").exists():
            with open(checkpoint_path / "training_state.json", "r") as f:
                training_state = json.load(f)
                self.global_step = training_state["global_step"]
                self.current_epoch = training_state["current_epoch"]
                self.best_val_loss = training_state["best_val_loss"]

        logger.info(f"Checkpoint loaded. Resuming from step {self.global_step}, epoch {self.current_epoch}")
