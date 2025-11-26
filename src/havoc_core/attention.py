from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    num_heads: int = 32
    head_dim: int = 128
    num_kv_heads: int = 8
    dropout: float = 0.0
    rotary_dim: Optional[int] = None
    rope_theta: float = 10000.0
    bias: bool = False
