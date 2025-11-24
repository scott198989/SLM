from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AttentionConfig:
    num_heads: int = 32
    head_dim: int = 128
    num_kv_heads: int = 8
    rotary_dim: Optional[int] = None
    rope_theta: float = 10000.0


@dataclass
class MLPConfig:
    hidden_dim: int = 11008
    activation: str = "swiglu"


@dataclass
class HavocConfig:
    vocab_size: int = 70000
    d_model: int = 4096
    num_layers: int = 32
    max_seq_len: int = 4096
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
    max_sequence_length: int = 4096


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
