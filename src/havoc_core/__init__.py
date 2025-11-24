"""HAVOC-7B Core Module

This module contains the core model architecture, configuration classes,
and tokenizer utilities for the HAVOC-7B transformer model.
"""

from havoc_core.config import (
    AttentionConfig,
    DataMixtureConfig,
    EvalConfig,
    HavocConfig,
    InferenceConfig,
    MLPConfig,
    RAGConfig,
    SRSConfig,
    TokenizerTrainingConfig,
    ToolConfig,
    TrainingConfig,
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
    "EvalConfig",
    "HavocConfig",
    "InferenceConfig",
    "MLPConfig",
    "RAGConfig",
    "SRSConfig",
    "TokenizerTrainingConfig",
    "ToolConfig",
    "TrainingConfig",
    # Model classes
    "HavocModel",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "TransformerBlock",
]
