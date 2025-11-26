from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    num_heads: int = 24
    head_dim: Optional[int] = None  # computed as d_model // num_heads if None
    num_kv_heads: int = 4
    dropout: float = 0.0
    rotary_dim: Optional[int] = None
    rope_theta: float = 10000.0
    bias: bool = False
