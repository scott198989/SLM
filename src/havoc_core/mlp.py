from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLPConfig:
    hidden_dim: int = 12288
    activation: str = "swiglu"
    dropout: float = 0.0
