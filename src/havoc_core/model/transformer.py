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

        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self._init_weights()

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
        batch_size, seq_len = input_ids.shape

        # Build causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)

        # Convert to additive attention mask for attention computation
        attention_mask = self._build_attention_mask(attention_mask, past_key_values)

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

    def generate(
        self, prompt_ids: torch.Tensor, max_new_tokens: int = 16, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using greedy decoding.

        Args:
            prompt_ids: Input prompt tokens [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = greedy)

        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated = prompt_ids
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # On first iteration, pass full prompt. On subsequent, only pass last token
                if past_key_values is None:
                    input_ids = generated
                else:
                    input_ids = generated[:, -1:]

                # Forward pass with KV-cache
                logits, past_key_values = self(
                    input_ids, past_key_values=past_key_values, use_cache=True
                )

                # Sample next token (greedy if temperature == 1.0)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we hit EOS token
                if (next_token == self.config.eos_token_id).all():
                    break

        return generated

    def _build_attention_mask(
        self,
        attention_mask: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Build causal attention mask for autoregressive decoding.

        Args:
            attention_mask: Binary mask [batch_size, seq_len] (1 = attend, 0 = ignore)
            past_key_values: Optional cached keys/values from previous forward passes

        Returns:
            Additive attention mask [batch_size, 1, seq_len, full_seq_len]
        """
        bsz, seq_len = attention_mask.shape

        # Determine the full sequence length (including cached tokens)
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]  # Shape: [batch, num_heads, past_len, head_dim]
            full_seq_len = past_len + seq_len
        else:
            full_seq_len = seq_len
            past_len = 0

        # Create causal mask: lower triangular for autoregressive attention
        # Shape: [seq_len, full_seq_len]
        causal_mask = torch.triu(
            torch.ones((seq_len, full_seq_len), device=attention_mask.device, dtype=torch.bool),
            diagonal=past_len + 1,
        )

        # Convert to additive mask: 0 for valid positions, -inf for masked positions
        # Shape: [batch_size, 1, seq_len, full_seq_len]
        additive_mask = torch.zeros((bsz, 1, seq_len, full_seq_len), device=attention_mask.device)
        additive_mask.masked_fill_(causal_mask, float("-inf"))

        return additive_mask

    def _init_weights(self) -> None:
        """Initialize weights using GPT-NeoX style initialization."""
        std = self.config.initializer_range

        # Initialize embeddings
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

        # Initialize layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy loss with label shifting for causal LM.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]. If None, uses input_ids shifted.
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Tuple of (loss, logits)
        """
        # Forward pass
        logits, _ = self(input_ids, attention_mask=attention_mask)

        # Compute loss
        if labels is None:
            # For causal LM, shift input_ids to create labels
            labels = input_ids

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss, logits

    def save_config(self, path: str) -> None:
        """Save model configuration to JSON."""
        import json
        import os
        from dataclasses import asdict

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

    def save_pretrained(self, save_directory: str, use_safetensors: bool = True) -> None:
        """
        Save model weights and configuration to a directory.

        Args:
            save_directory: Directory to save the model
            use_safetensors: Whether to use safetensors format (recommended)
        """
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        self.save_config(config_path)

        # Save model weights
        if use_safetensors:
            try:
                from safetensors.torch import save_file

                model_path = os.path.join(save_directory, "model.safetensors")
                save_file(self.state_dict(), model_path)
            except ImportError:
                print(
                    "safetensors not installed, falling back to PyTorch format. "
                    "Install with: pip install safetensors"
                )
                use_safetensors = False

        if not use_safetensors:
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)

    @classmethod
    def load_pretrained(
        cls, load_directory: str, device: Optional[str] = None, strict: bool = True
    ) -> "HavocModel":
        """
        Load model weights and configuration from a directory.

        Args:
            load_directory: Directory containing the saved model
            device: Device to load the model to (e.g., "cuda", "cpu")
            strict: Whether to strictly enforce state dict keys match

        Returns:
            Loaded HavocModel instance
        """
        import json
        import os
        from dataclasses import asdict

        # Load configuration
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Reconstruct config objects
        if "attention" in config_dict:
            from havoc_core.config import AttentionConfig

            config_dict["attention"] = AttentionConfig(**config_dict["attention"])
        if "mlp" in config_dict:
            from havoc_core.config import MLPConfig

            config_dict["mlp"] = MLPConfig(**config_dict["mlp"])

        config = HavocConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights - try safetensors first, then PyTorch format
        safetensors_path = os.path.join(load_directory, "model.safetensors")
        pytorch_path = os.path.join(load_directory, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(safetensors_path, device=device or "cpu")
                model.load_state_dict(state_dict, strict=strict)
            except ImportError:
                raise ImportError(
                    "safetensors is required to load this model. Install with: pip install safetensors"
                )
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location=device or "cpu")
            model.load_state_dict(state_dict, strict=strict)
        else:
            raise FileNotFoundError(
                f"No model weights found in {load_directory}. "
                f"Expected either model.safetensors or pytorch_model.bin"
            )

        if device:
            model = model.to(device)

        return model


SigmaModel = HavocModel
