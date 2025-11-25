from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from havoc_core.config import HavocConfig, InferenceConfig
from havoc_core.model.transformer import HavocModel
from havoc_core.tokenizer.tokenizer import HavocTokenizer, load_tokenizer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Inference engine for HAVOC-7B model.

    Handles:
    - Model loading from checkpoint
    - Text generation with various sampling strategies
    - Streaming generation
    - Batch inference
    - Device placement and mixed precision
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model = self._load_model()
        self.model.eval()
        self.tokenizer = self._create_tokenizer()

        logger.info(f"Inference engine initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

    def _load_model(self) -> HavocModel:
        """Load model from checkpoint or create new model."""
        if self.config.model_config is None:
            self.config.model_config = HavocConfig.havoc_7b()

        # Prefer safetensors if present
        checkpoint_path = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
        model = HavocModel(self.config.model_config)

        if checkpoint_path:
            resolved_model_path = None
            if checkpoint_path.is_dir():
                safetensors_path = checkpoint_path / "model.safetensors"
                pt_path = checkpoint_path / "model.pt"
                if safetensors_path.exists():
                    resolved_model_path = safetensors_path
                elif pt_path.exists():
                    resolved_model_path = pt_path
            else:
                resolved_model_path = checkpoint_path

            if resolved_model_path and resolved_model_path.exists():
                logger.info(f"Loading model from {resolved_model_path}")
                if resolved_model_path.suffix == ".safetensors":
                    try:
                        from safetensors.torch import load_file  # type: ignore

                        state_dict = load_file(str(resolved_model_path), device=str(self.device))
                        model.load_state_dict(state_dict)
                    except ImportError:
                        logger.warning(
                            "safetensors not installed; cannot load .safetensors checkpoint. "
                            "Install safetensors or provide a .pt checkpoint."
                        )
                else:
                    state_dict = torch.load(resolved_model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
            else:
                logger.warning(
                    "Checkpoint not found at %s. Using randomly initialized weights.",
                    resolved_model_path or checkpoint_path,
                )

        model.to(self.device)
        return model

    def _create_tokenizer(self):
        """Create tokenizer, falling back to a simple dummy tokenizer if artifacts are missing."""
        tok_path = getattr(self.config, "tokenizer_path", None)
        if tok_path and Path(tok_path).exists():
            try:
                logger.info("Loading tokenizer from %s", tok_path)
                return load_tokenizer(tok_path)
            except Exception as exc:  # pragma: no cover - best effort fallback
                logger.warning("Failed to load tokenizer at %s: %s. Falling back to dummy.", tok_path, exc)

        class DummyTokenizer:
            def __init__(self, vocab_size: int):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.bos_token_id = 1
                self.eos_token_id = 2
                self.unk_token_id = 3

            def encode(self, text: str) -> list[int]:
                tokens = [self.bos_token_id]
                for char in text[:100]:
                    tokens.append(min(ord(char) % self.vocab_size, self.vocab_size - 1))
                tokens.append(self.eos_token_id)
                return tokens

            def decode(self, tokens: list[int]) -> str:
                chars = []
                for token_id in tokens:
                    if token_id in (self.bos_token_id, self.pad_token_id):
                        continue
                    if token_id == self.eos_token_id:
                        break
                    chars.append(chr(token_id % 128))
                return "".join(chars)

        logger.warning("Using dummy tokenizer. Provide tokenizer artifacts at tokenizer_path for real decoding.")
        return DummyTokenizer(self.config.model_config.vocab_size)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy)
            stop_sequences: List of sequences that stop generation

        Returns:
            Generated text
        """
        # Use config defaults if not provided
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        )
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        output_ids = self._generate_tokens(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())

        # Check for stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]

        return generated_text

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """
        Generate text from a prompt with streaming output.

        Args:
            Same as generate()

        Yields:
            Generated text tokens one at a time
        """
        # Use config defaults if not provided
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        )
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate tokens one at a time
        past_key_values = None
        generated_tokens = []
        running_text = ""

        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                if self.config.use_amp:
                    dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
                    with autocast(dtype=dtype):
                        logits, past_key_values = self.model(
                            input_ids[:, -1:] if past_key_values is not None else input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                else:
                    logits, past_key_values = self.model(
                        input_ids[:, -1:] if past_key_values is not None else input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            # Get next token logits
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for token_id in set(generated_tokens):
                    next_token_logits[:, token_id] /= repetition_penalty

            # Sample next token
            next_token = self._sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Add to generated tokens
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Decode and yield
            token_text = self.tokenizer.decode([next_token.item()])
            running_text += token_text
            yield token_text

            # Check for stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq and stop_seq in running_text:
                        return

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """Internal token generation loop."""
        past_key_values = None
        generated_tokens = []

        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                if self.config.use_amp:
                    dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
                    with autocast(dtype=dtype):
                        logits, past_key_values = self.model(
                            input_ids[:, -1:] if past_key_values is not None else input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                else:
                    logits, past_key_values = self.model(
                        input_ids[:, -1:] if past_key_values is not None else input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            # Get next token logits
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for token_id in set(generated_tokens):
                    next_token_logits[:, token_id] /= repetition_penalty

            # Sample next token
            next_token = self._sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Add to generated tokens
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return input_ids

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if not do_sample:
            # Greedy sampling
            return torch.argmax(logits, dim=-1)

        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token
