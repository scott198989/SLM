from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig


@dataclass
class HavocConfig:
    vocab_size: int = 70000
    d_model: int = 2560
    num_layers: int = 20
    max_seq_len: int = 1024
    attention: AttentionConfig = field(
        default_factory=lambda: AttentionConfig(
            num_heads=32,
            num_kv_heads=4,
            head_dim=None,
            dropout=0.0,
            rotary_dim=None,
            rope_theta=10000.0,
            bias=False,
        )
    )
    mlp: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            hidden_dim=10240,
            activation="gelu",
            dropout=0.0,
        )
    )
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def havoc_2b(cls) -> "HavocConfig":
        """Return the default 2B architecture (Option-E)."""
        return cls()

    @classmethod
    def havoc_7b(cls) -> "HavocConfig":
        """
        Legacy alias to the current default (2B) config.

        For the full 7B specification, use Havoc7BConfig in config_7b.py.
        """
        return cls.havoc_2b()


@dataclass
class TokenizerTrainingConfig:
    """Configuration for SentencePiece tokenizer training.

    CANONICAL VOCAB SIZE: 70000
    This must match HavocConfig.vocab_size and Havoc7BConfig.vocab_size.
    """
    vocab_size: int = 70000  # CANONICAL: Must match model configs
    model_type: str = "bpe"
    special_tokens: List[str] = field(
        default_factory=lambda: [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "<tool>",
            "<ref>",
            "<plan>",
            "<exec>",
            "<argue>",
            "<arbiter>",
            "<audit>",
        ]
    )
    input_files: List[str] = field(default_factory=list)
    output_dir: str = "/workspace/SLM/artifacts/tokenizer"
    character_coverage: float = 0.9995
    max_sentence_length: int = 2048
    normalize_text: bool = True
    # Token IDs - MUST match SentencePiece training and model configs
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


@dataclass
class DataMixtureConfig:
    domain_ratio: float = 0.6
    general_ratio: float = 0.3
    dialog_ratio: float = 0.1
    max_sequence_length: int = 1024
    samples_per_epoch: int = 1024
    pack_sequences: bool = True
    add_bos: bool = True
    add_eos: bool = True


@dataclass
class ToolConfig:
    enable_python_math: bool = True
    enable_dsl: bool = True


@dataclass
class RAGConfig:
    embed_dim: int = 768
    index_factory: str = "FlatL2"
    top_k: int = 5
    cache_dir: Optional[str] = None


@dataclass
class SRSConfig:
    enable_arguments: bool = True
    enable_audit: bool = True
    rag: RAGConfig = field(default_factory=RAGConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    max_steps: int = 8


@dataclass
class EvalConfig:
    bench_name: str = "smoke"
    output_dir: str = "artifacts/eval"


@dataclass
class InferenceConfig:
    # Model
    model_config: Optional[HavocConfig] = None
    checkpoint_path: Optional[str] = None

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 8
    max_concurrent_requests: int = 100

    # Device
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Tokenizer
    tokenizer_path: str = "/workspace/SLM/artifacts/tokenizer"


@dataclass
class TrainingConfig:
    """Training configuration with stable defaults for 6-7B from-scratch pretraining.

    STABLE PHASE 1 DEFAULTS:
    - Conservative LR (1.5e-4) with long warmup (2500 steps)
    - Cosine decay to 10% of peak (1.5e-5)
    - Standard weight decay (0.1)
    - Gradient clipping at 1.0
    """
    # Model and data
    model_config: Optional[HavocConfig] = None
    data_config: Optional[DataMixtureConfig] = None
    tokenizer_path: Optional[str] = "/workspace/SLM/artifacts/tokenizer"
    data_sources: Optional[list] = None

    # Training hyperparameters - STABLE PHASE 1 DEFAULTS
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    max_epochs: Optional[int] = None  # Use max_steps for pretraining
    max_steps: Optional[int] = 100000
    learning_rate: float = 1.5e-4  # Conservative for 6B from-scratch
    weight_decay: float = 0.1  # Standard
    warmup_steps: int = 2500  # Long warmup for stability
    max_grad_norm: float = 1.0

    # Optimizer betas and eps (used by Trainer)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    min_learning_rate: float = 1.5e-5  # 10% of peak LR

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # Prefer bf16 on H100/H200/A100

    # Checkpointing (300GB volume - rotation required)
    checkpoint_dir: str = "/workspace/SLM/checkpoints"
    save_every_n_steps: int = 250  # Save every 250 steps
    keep_last_n_checkpoints: int = 4  # Keep last 4 only (space constraint)
    resume_from_checkpoint: Optional[str] = None

    # Validation
    eval_every_n_steps: int = 500
    eval_samples: int = 100
    log_eval_examples: int = 1
    example_prompt_length: int = 64
    example_max_new_tokens: int = 32

    # Logging
    log_every_n_steps: int = 10
    log_dir: str = "/workspace/SLM/logs"
    log_json_metrics: bool = True
    use_tensorboard: bool = False

    # Device
    device: str = "cuda"
    seed: int = 42
