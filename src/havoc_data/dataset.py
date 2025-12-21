from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from havoc_core.config import DataMixtureConfig
from havoc_data.preprocess import normalize_text, iter_normalized
from havoc_data.sources import DataSource


@dataclass
class SequenceExample:
    """A single training example."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class CausalLMDataset(IterableDataset):
    """Streaming causal language model dataset with mixture support.

    Features:
    - Streaming from multiple data sources
    - Mixture weighting
    - Optional sample packing
    - Proper BOS/EOS token handling
    - Document-level preprocessing

    Args:
        tokenizer: Tokenizer instance with encode/decode methods
        sources: List of DataSource objects
        mixture: DataMixtureConfig with mixture settings
        max_seq_len: Maximum sequence length
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        enable_packing: Whether to pack multiple documents into sequences
        preprocessing_config: Optional preprocessing settings
    """

    def __init__(
        self,
        tokenizer,
        sources: List[DataSource],
        mixture: DataMixtureConfig,
        max_seq_len: int = 4096,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        enable_packing: bool = False,
        extract_dsl: bool = True,
        annotate_reasoning: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sources = sources
        self.mixture = mixture
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.enable_packing = enable_packing
        self.extract_dsl = extract_dsl
        self.annotate_reasoning = annotate_reasoning

        # Validate sources have weights
        if not all(hasattr(s, "weight") and s.weight > 0 for s in sources):
            raise ValueError("All sources must have positive weights")

        # Create mixture policy
        self.mixture_policy = MixturePolicy(sources)

    def __iter__(self) -> Iterator[SequenceExample]:
        """Iterate over training examples.

        Yields:
            SequenceExample instances with input_ids, attention_mask, and labels
        """
        if self.enable_packing:
            yield from self._iter_packed()
        else:
            yield from self._iter_unpacked()

    def _iter_unpacked(self) -> Iterator[SequenceExample]:
        """Iterate without packing - one document per sequence."""
        while True:
            # Sample a source according to mixture weights
            source = self.mixture_policy.sample()

            # Get documents from the source
            for doc in source.iter_documents():
                # Preprocess text
                text = doc["text"]
                text = normalize_text(
                    text,
                    extract_dsl=self.extract_dsl,
                    annotate_reasoning=self.annotate_reasoning,
                )

                # Tokenize
                try:
                    # Most tokenizers expect a method like encode()
                    if hasattr(self.tokenizer, "encode"):
                        tokens = self.tokenizer.encode(text)
                    else:
                        # Fallback for custom tokenizers
                        tokens = self.tokenizer(text)

                    # Add BOS/EOS
                    tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

                    # Truncate if needed
                    if len(tokens) > self.max_seq_len:
                        tokens = tokens[: self.max_seq_len]

                    # Convert to tensors
                    input_ids = torch.tensor(tokens, dtype=torch.long)

                    # Create attention mask (1 for real tokens, 0 for padding)
                    attention_mask = torch.ones_like(input_ids)

                    # Pad to max length
                    if len(tokens) < self.max_seq_len:
                        pad_len = self.max_seq_len - len(tokens)
                        input_ids = torch.cat(
                            [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
                        )
                        attention_mask = torch.cat(
                            [attention_mask, torch.zeros(pad_len, dtype=torch.long)]
                        )

                    # Labels are the same as input_ids for causal LM
                    # But we mask out padding tokens with -100
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100

                    yield SequenceExample(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )

                except Exception as e:
                    # Skip documents that fail to tokenize
                    print(f"Warning: Failed to process document from {source.name}: {e}")
                    continue

    def _iter_packed(self) -> Iterator[SequenceExample]:
        """Iterate with packing - multiple documents per sequence.

        This packs multiple shorter documents into a single sequence to reduce padding.
        Documents are separated by EOS tokens.
        """
        buffer = [self.bos_token_id]  # Start with BOS
        doc_iterator = self._get_document_iterator()

        for doc in doc_iterator:
            # Preprocess
            text = doc["text"]
            text = normalize_text(
                text, extract_dsl=self.extract_dsl, annotate_reasoning=self.annotate_reasoning
            )

            # Tokenize
            try:
                if hasattr(self.tokenizer, "encode"):
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = self.tokenizer(text)

                # Add EOS after each document
                tokens = tokens + [self.eos_token_id]

                # Check if adding this document would exceed max length
                if len(buffer) + len(tokens) > self.max_seq_len:
                    # Emit current buffer
                    if len(buffer) > 1:  # More than just BOS
                        yield self._create_example(buffer)

                    # Start new buffer
                    buffer = [self.bos_token_id] + tokens
                else:
                    # Add to buffer
                    buffer.extend(tokens)

                # If buffer is full, emit it
                if len(buffer) >= self.max_seq_len:
                    yield self._create_example(buffer)
                    buffer = [self.bos_token_id]

            except Exception as e:
                print(f"Warning: Failed to process document: {e}")
                continue

    def _get_document_iterator(self) -> Iterator[dict]:
        """Get an infinite iterator over documents from all sources."""
        while True:
            source = self.mixture_policy.sample()
            for doc in source.iter_documents():
                yield doc

    def _create_example(self, tokens: List[int]) -> SequenceExample:
        """Create a SequenceExample from a list of tokens."""
        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Pad if needed
        if len(tokens) < self.max_seq_len:
            pad_len = self.max_seq_len - len(tokens)
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
            )
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

        # Create labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return SequenceExample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class MixturePolicy:
    """Policy for sampling from multiple data sources according to mixture weights."""

    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self.weights = [s.weight for s in sources]

        # Normalize weights to sum to 1
        total = sum(self.weights)
        self.normalized_weights = [w / total for w in self.weights]

    def sample(self) -> DataSource:
        """Sample a data source according to mixture weights.

        Returns:
            A DataSource instance
        """
        return random.choices(self.sources, weights=self.weights, k=1)[0]

    def get_mixture_stats(self) -> dict:
        """Get statistics about the mixture.

        Returns:
            Dict with source names and their sampling probabilities
        """
        return {
            source.name: weight for source, weight in zip(self.sources, self.normalized_weights)
        }


def causal_lm_collate_fn(batch: List[SequenceExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for CausalLMDataset.

    Converts a list of SequenceExample objects to the format expected by Trainer.

    Args:
        batch: List of SequenceExample objects

    Returns:
        Tuple of (input_ids, attention_mask) tensors with shape (batch_size, seq_len)
    """
    input_ids = torch.stack([ex.input_ids for ex in batch])
    attention_mask = torch.stack([ex.attention_mask for ex in batch])
    return input_ids, attention_mask


def causal_lm_collate_fn_with_labels(
    batch: List[SequenceExample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that includes labels.

    Args:
        batch: List of SequenceExample objects

    Returns:
        Tuple of (input_ids, attention_mask, labels)
    """
    input_ids = torch.stack([ex.input_ids for ex in batch])
    attention_mask = torch.stack([ex.attention_mask for ex in batch])
    labels = torch.stack([ex.labels for ex in batch])
    return input_ids, attention_mask, labels
