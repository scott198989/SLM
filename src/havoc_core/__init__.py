"""HAVOC-7B Core Module

This module contains the core model architecture, configuration classes,
and tokenizer utilities for the HAVOC-7B transformer model.
"""

from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig
from havoc_core.config import (
    HavocConfig,
    DataMixtureConfig,
    TokenizerTrainingConfig,
    TrainingConfig,
    EvalConfig,
    InferenceConfig,
    RAGConfig,
    SRSConfig,
    ToolConfig,
)
from havoc_core.model.transformer import HavocModel
from havoc_core.model.blocks import (
    RMSNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    SwiGLU,
    TransformerBlock,
)

__all__ = [
    # Config classes
    "AttentionConfig",
    "DataMixtureConfig",
    "HavocConfig",
    "MLPConfig",
    "TokenizerTrainingConfig",
    "TrainingConfig",
    "EvalConfig",
    "InferenceConfig",
    "RAGConfig",
    "SRSConfig",
    "ToolConfig",
    # Model classes
    "HavocModel",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "TransformerBlock",
]
