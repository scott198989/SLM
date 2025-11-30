from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        # eps may come in as a string from YAML; ensure numeric for stability
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = x.new_tensor(self.eps)  # keep eps on the right device/dtype
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (rotate_half(x) * sin)


@dataclass
class AttentionOutput:
    attn_output: torch.Tensor
    present_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class GQAttention(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        attn_cfg: AttentionConfig = config.attention
        self.num_heads = attn_cfg.num_heads
        self.num_kv_heads = attn_cfg.num_kv_heads
        self.head_dim = attn_cfg.head_dim or (config.d_model // attn_cfg.num_heads)
        self.hidden_size = config.d_model

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        rope_dim = attn_cfg.rotary_dim or self.head_dim
        self.rotary = RotaryEmbedding(dim=rope_dim, base=attn_cfg.rope_theta)

    def _shape(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        new_shape = x.size()[:-1] + (num_heads, self.head_dim)
        return x.view(*new_shape).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        bsz, seq_len, _ = hidden_states.size()
        q = self._shape(self.q_proj(hidden_states), self.num_heads)
        k = self._shape(self.k_proj(hidden_states), self.num_kv_heads)
        v = self._shape(self.v_proj(hidden_states), self.num_kv_heads)

        # Determine position offset for RoPE when using KV-cache
        device = hidden_states.device
        if past_key_value is not None:
            past_len = past_key_value[0].shape[2]
        else:
            past_len = 0

        # Generate RoPE embeddings for the full sequence length
        full_seq_len = past_len + seq_len
        rope_cos_sin = self.rotary(full_seq_len, device=device)

        # Extract cos/sin for current positions
        cos = rope_cos_sin.cos()[past_len : past_len + seq_len, : self.head_dim][None, None, :, :]
        sin = rope_cos_sin.sin()[past_len : past_len + seq_len, : self.head_dim][None, None, :, :]

        # Apply RoPE to query and key
        q_rot = self.rotary.apply_rotary(q[..., : self.rotary.dim], cos, sin)
        k_rot = self.rotary.apply_rotary(k[..., : self.rotary.dim], cos, sin)

        # Concatenate rotated and non-rotated parts
        q = torch.cat([q_rot, q[..., self.rotary.dim :]], dim=-1)
        k = torch.cat([k_rot, k[..., self.rotary.dim :]], dim=-1)

        # Concatenate with past keys/values if using cache
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        if use_cache:
            present = (k, v)
        else:
            present = None

        # Expand KV heads for GQA
        k_expanded = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_expanded = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Compute attention scores
        attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.out_proj(attn_output)
        return AttentionOutput(attn_output=output, present_key_value=present)


class SwiGLU(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        hidden_dim = config.mlp.hidden_dim
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.activation = getattr(config.mlp, "activation", "swiglu").lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            hidden = F.silu(self.w1(x)) * self.w2(x)
        else:
            act_fn = getattr(F, self.activation, F.gelu)
            hidden = act_fn(self.w1(x)) * self.w2(x)
        return self.w3(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = GQAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attn_out.attn_output)

        mlp_out = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + self.dropout(mlp_out)
        return hidden_states, attn_out.present_key_value


# Alias for compatibility
GroupedQueryAttention = GQAttention
