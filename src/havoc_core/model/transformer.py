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

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Init
        self._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        embeddings = self.embed_tokens(input_ids)
        hidden_states = embeddings
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)

        attention_mask = self._build_attention_mask(attention_mask, past_key_values)

        new_key_values = []
        for layer, past in zip(self.layers, past_key_values):
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

    # ---------------------------------------------------------
    # FIXED generate() — real sampling + dot penalty
    # ---------------------------------------------------------
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        tokenizer=None,   # added tokenizer for penalty logic
    ):
        self.eval()
        generated = prompt_ids
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):

                if past_key_values is None:
                    input_ids = generated
                else:
                    input_ids = generated[:, -1:]

                logits, past_key_values = self(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)

                # ------------------------------
                # FORCE STOP SELECTING "."
                # ------------------------------
                if tokenizer is not None:
                    dot_id = tokenizer.encode(".")[0]
                    probs[:, dot_id] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # multinomial sample
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == self.config.eos_token_id).all():
                    break

        return generated

    # END OF generate()

    # ---------------------------------------------------------
    # Mask + weight init functions unchanged
    # ---------------------------------------------------------
    def _build_attention_mask(self, attention_mask, past_key_values=None):
        bsz, seq_len = attention_mask.shape

        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
            full_seq_len = past_len + seq_len
        else:
            past_len = 0
            full_seq_len = seq_len

        causal_mask = torch.triu(
            torch.ones((seq_len, full_seq_len), device=attention_mask.device, dtype=torch.bool),
            diagonal=past_len + 1,
        )

        additive_mask = torch.zeros((bsz, 1, seq_len, full_seq_len), device=attention_mask.device)
        additive_mask.masked_fill_(causal_mask, float("-inf"))

        return additive_mask

    def _init_weights(self):
        std = self.config.initializer_range

        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
