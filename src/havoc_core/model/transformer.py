from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from havoc_core.config import HavocConfig
from havoc_core.model.blocks import RMSNorm, TransformerBlock


class HavocModel(nn.Module):
    def __init__(self, config: HavocConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @classmethod
    def from_config(cls, config: HavocConfig) -> "HavocModel":
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        embeddings = self.embed_tokens(input_ids)
        hidden_states = embeddings

        new_key_values = []
        for i, (layer, past) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
            )
            new_key_values.append(present)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, (new_key_values if use_cache else None)

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 16) -> torch.Tensor:
        self.eval()
        generated = prompt_ids
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(generated, dtype=generated.dtype, device=generated.device)
            logits, past_key_values = self(
                generated[:, -self.config.max_seq_len :],
                attention_mask=self._build_attention_mask(attention_mask),
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def _build_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # Convert 1/0 attention_mask into additive mask for causal attention
        bsz, seq_len = attention_mask.shape
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=attention_mask.device))
        combined = attention_mask[:, None, None, :] * causal_mask
        mask = torch.where(combined > 0, 0.0, -1e9)
        return mask

    def save_config(self, path: str) -> None:
        # TODO: integrate with robust serialization if needed
        import json
        from dataclasses import asdict

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)


SigmaModel = HavocModel
