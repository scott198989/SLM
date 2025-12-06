"""
Memory-Optimized Trainer for HAVOC-7B

Supports single-GPU and multi-GPU (DDP) training
Uses gradient checkpointing, mixed precision, and gradient accumulation
"""

from __future__ import annotations

import os
import json
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import asdict

from havoc_core.config_7b import Havoc7BConfig, OptimizedTrainingConfig
from havoc_core.model.prime_model import HavocPrimeModel


class OptimizedTrainer:
    """
    Memory-optimized trainer for HAVOC-7B on RTX 5090

    Features:
    - Gradient checkpointing (CRITICAL for 24GB)
    - Mixed precision (bfloat16)
    - Gradient accumulation
    - FlashAttention support
    - Checkpoint save/load/resume
    - Validation with perplexity
    """

    def __init__(
        self,
        model: HavocPrimeModel,
        train_config: OptimizedTrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None
    ):
        self.model = model
        self.config = train_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer

        # Distributed training setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = (not self.is_distributed) or (self.local_rank == 0)

        # Device setup
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Enable gradient checkpointing
        if train_config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Wrap model in DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            if self.is_main_process:
                print(f"✓ Model wrapped in DistributedDataParallel (world_size={self.world_size})")

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler
        if train_config.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Create directories (only on main process)
        if self.is_main_process:
            Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)

        # Print memory estimate (only on main process)
        if self.is_main_process:
            print("\n" + "=" * 70)
            print("INITIALIZING OPTIMIZED TRAINER FOR HAVOC-7B")
            print("=" * 70)
            train_config.print_memory_estimate()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        print("Enabling gradient checkpointing...")

        # Enable PyTorch's gradient checkpointing on the base model
        if hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
            self.model.base_model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled via built-in method")
        else:
            # Fallback: manually wrap transformer layers
            from torch.utils.checkpoint import checkpoint

            # Store original forward methods
            original_forwards = []
            for layer in self.model.base_model.layers:
                original_forwards.append(layer.forward)

            # Wrap forward passes with checkpointing
            for i, layer in enumerate(self.model.base_model.layers):
                if i % self.config.checkpoint_every_n_layers == 0:
                    original_forward = layer.forward

                    def create_checkpointed_forward(original_fn):
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint(original_fn, *args, **kwargs, use_reentrant=False)
                        return checkpointed_forward

                    layer.forward = create_checkpointed_forward(original_forward)

            print(f"  Checkpointing every {self.config.checkpoint_every_n_layers} layers (manual wrapper)")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay"""

        # Separate parameters into weight decay and no weight decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for biases and layer norms
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        # Use fused AdamW if available (faster)
        if self.config.optimizer == "adamw_fused":
            try:
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                    fused=True
                )
                print("Using fused AdamW optimizer")
            except:
                # Fall back to regular AdamW
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon
                )
                print("Using standard AdamW optimizer (fused not available)")
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
            print("Using standard AdamW optimizer")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler"""

        if self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps - self.config.warmup_steps,
                eta_min=self.config.min_learning_rate
            )
            print(f"Using cosine LR scheduler (min_lr={self.config.min_learning_rate})")

        elif self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR

            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_learning_rate / self.config.learning_rate,
                total_iters=self.config.max_steps
            )
            print("Using linear LR scheduler")

        else:
            from torch.optim.lr_scheduler import ConstantLR
            scheduler = ConstantLR(self.optimizer, factor=1.0)
            print("Using constant LR scheduler")

        return scheduler

    def train(self):
        """Main training loop"""

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params_billions():.2f}B")
        print(f"Effective batch size: {self.config.get_effective_batch_size()}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Gradient checkpointing: {self.config.gradient_checkpointing}")
        print(f"Mixed precision: {self.config.use_amp} ({self.config.amp_dtype})")
        print("=" * 70 + "\n")

        self.model.train()

        while self.global_step < self.config.max_steps:
            epoch_loss = self._train_epoch()

            self.current_epoch += 1

            print(f"Epoch {self.current_epoch} completed | Avg Loss: {epoch_loss:.4f}")

            # Check if max steps reached
            if self.global_step >= self.config.max_steps:
                break

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total steps: {self.global_step}")
        print(f"Total epochs: {self.current_epoch}")
        print("=" * 70 + "\n")

    def _train_epoch(self) -> float:
        """Train for one epoch"""

        epoch_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx == 0:
                print(f"[DEBUG] Loading first batch (batch_size={self.config.batch_size})...")

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            if batch_idx == 0:
                print(f"[DEBUG] First batch loaded. Starting forward pass...")

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.config.use_amp, dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16):
                logits, _ = self.model(input_ids)

                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step (after warmup)
                if self.global_step >= self.config.warmup_steps:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    self._log_step(loss.item() * self.config.gradient_accumulation_steps)

                # Validation
                if self.global_step % self.config.eval_every_n_steps == 0 and self.val_dataloader:
                    val_loss = self._validate()
                    self.model.train()

                # Checkpointing
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint()

                # Check max steps
                if self.global_step >= self.config.max_steps:
                    break

        return epoch_loss / max(num_batches, 1)

    def _log_step(self, loss: float):
        """Log training step"""
        lr = self.optimizer.param_groups[0]['lr']

        print(
            f"Step {self.global_step:6d} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Epoch: {self.current_epoch}"
        )

    def _validate(self) -> float:
        """Run validation"""
        print("\nRunning validation...")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                logits, _ = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= self.config.eval_samples:
                    break

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}\n")

        return avg_loss

    def _save_checkpoint(self):
        """Save training checkpoint"""

        checkpoint_dir = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint to {checkpoint_dir}...")

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save optimizer
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save scheduler
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # Save scaler
        if self.scaler:
            torch.save(self.scaler.state_dict(), checkpoint_dir / "scaler.pt")

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss
        }
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)

        print(f"Checkpoint saved!")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N"""

        checkpoint_parent = Path(self.config.checkpoint_dir)
        checkpoints = sorted(checkpoint_parent.glob("checkpoint_step_*"), key=os.path.getmtime)

        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_last_n_checkpoints]:
                print(f"Removing old checkpoint: {checkpoint}")
                import shutil
                shutil.rmtree(checkpoint)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training"""

        checkpoint_dir = Path(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_dir}...")

        # Load model
        state_dict = torch.load(checkpoint_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Load optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        # Load scheduler
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))

        # Load scaler
        scaler_path = checkpoint_dir / "scaler.pt"
        if scaler_path.exists() and self.scaler:
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                training_state = json.load(f)
                self.global_step = training_state["global_step"]
                self.current_epoch = training_state["current_epoch"]
                self.best_val_loss = training_state["best_val_loss"]

        print(f"Checkpoint loaded! Resuming from step {self.global_step}")
