from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig


@dataclass
class HavocConfig:
    vocab_size: int = 70000
    d_model: int = 3072
    num_layers: int = 22
    max_seq_len: int = 2048
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def havoc_7b(cls) -> "HavocConfig":
        return cls()


@dataclass
class TokenizerTrainingConfig:
    vocab_size: int = 75000
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
    output_dir: str = "artifacts/tokenizer"
    character_coverage: float = 0.9995
    max_sentence_length: int = 2048
    normalize_text: bool = True


@dataclass
class DataMixtureConfig:
    domain_ratio: float = 0.6
    general_ratio: float = 0.3
    dialog_ratio: float = 0.1
    max_sequence_length: int = 2048
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
    tokenizer_path: str = "artifacts/tokenizer"


@dataclass
class TrainingConfig:
    # Model and data
    model_config: Optional[HavocConfig] = None
    data_config: Optional[DataMixtureConfig] = None
    tokenizer_path: Optional[str] = None
    data_sources: Optional[list] = None

    # Training hyperparameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: Optional[int] = 10
    max_steps: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    min_learning_rate: float = 3e-5

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "bfloat16" or "float16"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Validation
    eval_every_n_steps: int = 500
    eval_samples: int = 100
    log_eval_examples: int = 1
    example_prompt_length: int = 64
    example_max_new_tokens: int = 32

    # Logging
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    log_json_metrics: bool = True
    use_tensorboard: bool = False

    # Device
    device: str = "cuda"
    seed: int = 42
