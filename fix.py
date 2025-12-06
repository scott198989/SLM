import os

# The corrected code for optimized_trainer.py
trainer_code = r'''"""
Memory-Optimized Trainer for HAVOC-7B
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
from typing import Optional, Dict, Any
from pathlib import Path
import signal

from havoc_core.config_7b import Havoc7BConfig, OptimizedTrainingConfig
from havoc_core.model.prime_model import HavocPrimeModel

class OptimizedTrainer:
    """
    Memory-optimized trainer for HAVOC-7B
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

        # Distributed setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = (not self.is_distributed) or (self.local_rank == 0)

        # --- THE FIX: ROBUST DEVICE SELECTION ---
        # Instead of guessing the device index, we trust the launch script.
        # We simply grab whatever GPU is currently active for this process.
        if torch.cuda.is_available():
            current_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{current_idx}")
        else:
            self.device = torch.device("cpu")

        # Move model to device
        self.model.to(self.device)

        # Gradient Checkpointing
        if train_config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # DDP Wrapper
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False
            )
            if self.is_main_process:
                print(f"✓ Model wrapped in DDP (world_size={self.world_size})")

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        if train_config.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        if self.is_main_process:
            Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print("INITIALIZING OPTIMIZED TRAINER FOR HAVOC-7B")

    def _enable_gradient_checkpointing(self):
        if hasattr(self.model, "module"):
             base = self.model.module.base_model
        else:
             base = self.model.base_model

        if hasattr(base, 'gradient_checkpointing_enable'):
            base.gradient_checkpointing_enable()
            if self.is_main_process:
                print("  Gradient checkpointing enabled")
        else:
            if self.is_main_process:
                print("  Using manual checkpointing wrapper")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        for name, param in model_ref.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        return torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": self.config.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )

    def _create_scheduler(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=self.config.min_learning_rate
        )

    def train(self):
        self.model.train()

        # --- ROBUST SIGNAL HANDLING ---
        import signal
        interrupted = False
        def signal_handler(sig, frame):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                print(f"\n[Rank {self.local_rank}] INTERRUPT SIGNAL RECEIVED")

        if self.is_main_process:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        if self.is_main_process:
            print(f"STARTING TRAINING on {self.device}")

        try:
            while self.global_step < self.config.max_steps and not interrupted:
                epoch_loss = self._train_epoch()
                self.current_epoch += 1

                if self.is_main_process:
                    print(f"Epoch {self.current_epoch} completed | Avg Loss: {epoch_loss:.4f}")

                if self.global_step >= self.config.max_steps or interrupted:
                    break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if interrupted and self.is_main_process:
                print("\nSaving emergency checkpoint...")
                self._save_checkpoint()

        if self.is_main_process:
            print("TRAINING FINISHED")

    def _train_epoch(self) -> float:
        epoch_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            with torch.amp.autocast('cuda', enabled=self.config.use_amp, dtype=torch.bfloat16):
                logits, _ = self.model(input_ids)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                if self.global_step >= self.config.warmup_steps:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.log_every_n_steps == 0 and self.is_main_process:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {self.global_step:6d} | Loss: {(loss.item() * self.config.gradient_accumulation_steps):.4f} | LR: {lr:.2e}")

                if self.global_step % self.config.save_every_n_steps == 0 and self.is_main_process:
                    self._save_checkpoint()

                if self.global_step >= self.config.max_steps:
                    break

        return epoch_loss / max(num_batches, 1)

    def _save_checkpoint(self):
        checkpoint_dir = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to {checkpoint_dir}...")

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(str(checkpoint_dir))

        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        checkpoint_parent = Path(self.config.checkpoint_dir)
        checkpoints = sorted(checkpoint_parent.glob("checkpoint_step_*"), key=os.path.getmtime)
        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for cp in checkpoints[:-self.config.keep_last_n_checkpoints]:
                import shutil
                shutil.rmtree(cp)

    def load_checkpoint(self, path):
        pass
'''

# WRITE THE FILE
file_path = "src/havoc_training/optimized_trainer.py"
print(f"Overwriting {file_path} with robust logic...")
with open(file_path, "w") as f:
    f.write(trainer_code)
print("Success. OptimizedTrainer is now patched.")
