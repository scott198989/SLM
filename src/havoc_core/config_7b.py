"""
HAVOC-7B Configuration

Extends base HavocConfig with 7B-specific settings and PRIME integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from havoc_core.config import HavocConfig
from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig


@dataclass
class Havoc7BConfig(HavocConfig):
    """
    HAVOC-7B Production Configuration

    7 billion parameter decoder-only transformer with:
    - 32 layers
    - d_model: 4096
    - 32 attention heads (8 KV heads for GQA)
    - SwiGLU MLP with ~2.7x expansion (11008 hidden dim)
    - RoPE positional embeddings
    - RMSNorm
    - FlashAttention-2 support

    Total parameters: ~6.96B ≈ 7B
    """

    # Model architecture
    vocab_size: int = 70000
    d_model: int = 4096
    num_layers: int = 32
    max_seq_len: int = 4096

    # Attention configuration
    attention: AttentionConfig = field(default_factory=lambda: AttentionConfig(
        num_heads=32,
        num_kv_heads=8,  # GQA ratio 4:1
        head_dim=128,  # d_model / num_heads = 4096 / 32
        dropout=0.0,
        rotary_dim=128,
        rope_theta=10000.0,  # Standard RoPE base frequency
        bias=False
    ))

    # MLP configuration
    mlp: MLPConfig = field(default_factory=lambda: MLPConfig(
        hidden_dim=11008,  # ~2.7x expansion for SwiGLU
        activation="swiglu",
        bias=False
    ))

    # Regularization
    dropout: float = 0.0  # No dropout in production models
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Standard special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # Reasoning tokens (for chain-of-thought)
    reason_start_token_id: int = 10
    reason_end_token_id: int = 11
    tool_start_token_id: int = 12
    tool_end_token_id: int = 13
    advocate_token_id: int = 14
    attack_token_id: int = 16
    pragmatist_token_id: int = 18

    # PRIME configuration
    enable_prime: bool = True  # Enable PRIME meta-reasoning by default
    prime_budget_auto: bool = True  # Auto-determine budget from task

    # Training optimization flags
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False  # Set True during training
    checkpoint_every_n_layers: int = 4

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "Havoc7BConfig":
        """Load config from checkpoint"""
        import json
        from pathlib import Path

        config_file = Path(checkpoint_path) / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        else:
            # Return default 7B config
            return cls()

    def save_pretrained(self, save_path: str):
        """Save config to checkpoint"""
        import json
        from pathlib import Path

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_file = save_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def get_param_count(self) -> int:
        """Calculate total parameter count"""
        # Embedding
        embedding_params = self.vocab_size * self.d_model

        # Attention per layer
        q_params = self.d_model * (self.attention.num_heads * self.attention.head_dim)
        kv_params = self.d_model * (self.attention.num_kv_heads * self.attention.head_dim) * 2
        o_params = (self.attention.num_heads * self.attention.head_dim) * self.d_model
        attn_params_per_layer = q_params + kv_params + o_params

        # MLP per layer
        mlp_params_per_layer = (
            self.d_model * self.mlp.hidden_dim +  # w1
            self.d_model * self.mlp.hidden_dim +  # w2
            self.mlp.hidden_dim * self.d_model    # w3
        )

        # RMSNorm per layer (2 per layer: attn_norm + mlp_norm)
        norm_params_per_layer = 2 * self.d_model

        # Total per layer
        params_per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer

        # Total model
        total_params = (
            embedding_params +  # Input embeddings
            params_per_layer * self.num_layers +  # Transformer layers
            self.d_model  # Final RMSNorm
            # Output layer is weight-tied, so no additional params
        )

        return total_params

    def get_param_count_billions(self) -> float:
        """Get parameter count in billions"""
        return self.get_param_count() / 1e9

    def __repr__(self) -> str:
        params_b = self.get_param_count_billions()
        return (
            f"Havoc7BConfig(\n"
            f"  parameters={params_b:.2f}B,\n"
            f"  layers={self.num_layers},\n"
            f"  d_model={self.d_model},\n"
            f"  heads={self.attention.num_heads},\n"
            f"  kv_heads={self.attention.num_kv_heads},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  prime_enabled={self.enable_prime}\n"
            f")"
        )


@dataclass
class ReasoningTokenConfig:
    """Configuration for reasoning token system"""

    # Token mappings (must match tokenizer)
    token_to_id: Dict[str, int] = field(default_factory=lambda: {
        "<reason>": 10,
        "</reason>": 11,
        "<tool>": 12,
        "</tool>": 13,
        "<advocate>": 14,
        "</advocate>": 15,
        "<attack>": 16,
        "</attack>": 17,
        "<pragmatist>": 18,
        "</pragmatist>": 19,
    })

    # Generation settings
    enable_reasoning_tokens: bool = True
    strip_reasoning_from_final: bool = False  # Keep reasoning visible

    def get_token_id(self, token: str) -> Optional[int]:
        """Get token ID by name"""
        return self.token_to_id.get(token)

    def get_token_name(self, token_id: int) -> Optional[str]:
        """Get token name by ID"""
        id_to_token = {v: k for k, v in self.token_to_id.items()}
        return id_to_token.get(token_id)


@dataclass
class OptimizedTrainingConfig:
    """
    Memory-optimized training configuration for RTX 5090 (24GB VRAM)

    This config is designed to fit 7B model training in 24GB VRAM using:
    - Mixed precision (bfloat16)
    - Gradient accumulation
    - Gradient checkpointing
    - Flash Attention
    """

    # Model
    model_config: Havoc7BConfig = field(default_factory=Havoc7BConfig)

    # Batch size
    batch_size: int = 1  # CRITICAL: Keep at 1 for 24GB GPU
    gradient_accumulation_steps: int = 32  # Effective batch = 32

    # Sequence length
    max_seq_len: int = 2048  # Start with 2048, can extend to 4096 later

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # RTX 5090 supports bf16 natively

    # Gradient checkpointing (CRITICAL for memory)
    gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 4  # Checkpoint every 4 layers

    # Flash Attention (CRITICAL for memory)
    use_flash_attention: bool = True

    # CPU offloading (optional, makes training slower but saves memory)
    cpu_offload_optimizer: bool = False  # Set True if OOM
    cpu_offload_params: bool = False

    # Optimizer
    optimizer: str = "adamw_fused"  # Fused AdamW for efficiency
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    max_steps: int = 100000

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints/havoc_7b"
    save_every_n_steps: int = 5000
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Validation
    eval_every_n_steps: int = 1000
    eval_samples: int = 200

    # Logging
    log_every_n_steps: int = 10
    log_dir: str = "logs/havoc_7b"

    # Device
    device: str = "cuda"
    seed: int = 42

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        return self.batch_size * self.gradient_accumulation_steps

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in GB"""
        params = self.model_config.get_param_count()

        # Model weights (bf16)
        model_memory = params * 2 / 1e9

        # Gradients (bf16)
        grad_memory = params * 2 / 1e9

        # Optimizer states (fp32, but can be offloaded)
        if self.cpu_offload_optimizer:
            optimizer_memory = 0  # On CPU
        else:
            optimizer_memory = params * 8 / 1e9  # AdamW: 2 states × 4 bytes

        # Activations (rough estimate, reduced by gradient checkpointing)
        if self.gradient_checkpointing:
            activation_memory = 2.0  # Much reduced
        else:
            activation_memory = 4.0

        # Misc buffers
        misc_memory = 1.0

        total = (
            model_memory +
            grad_memory +
            optimizer_memory +
            activation_memory +
            misc_memory
        )

        return {
            "model": model_memory,
            "gradients": grad_memory,
            "optimizer": optimizer_memory,
            "activations": activation_memory,
            "misc": misc_memory,
            "total": total
        }

    def print_memory_estimate(self):
        """Print memory usage estimate"""
        mem = self.estimate_memory_usage()
        print("=" * 60)
        print("MEMORY USAGE ESTIMATE (GB)")
        print("=" * 60)
        print(f"Model weights (bf16):    {mem['model']:6.2f} GB")
        print(f"Gradients (bf16):        {mem['gradients']:6.2f} GB")
        print(f"Optimizer states:        {mem['optimizer']:6.2f} GB")
        print(f"Activations:             {mem['activations']:6.2f} GB")
        print(f"Misc buffers:            {mem['misc']:6.2f} GB")
        print("-" * 60)
        print(f"TOTAL:                   {mem['total']:6.2f} GB")
        print("=" * 60)

        if mem['total'] > 24:
            print("⚠️  WARNING: Estimated memory exceeds 24GB!")
            print("   Consider:")
            print("   - Enabling cpu_offload_optimizer")
            print("   - Reducing max_seq_len")
            print("   - Increasing checkpoint_every_n_layers")
        else:
            print("✅ Memory estimate fits in RTX 5090 (24GB)")
