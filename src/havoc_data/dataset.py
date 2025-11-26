from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Dataset

from havoc_core.config import DataMixtureConfig
from havoc_data.preprocess import iter_normalized
from havoc_data.sources import DataSource


@dataclass
class SequenceExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class MixturePolicy:
    """Simple weighted sampler for data sources."""

    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self.weights = [max(s.weight, 0.0) for s in sources]
        # Avoid zero-sum
        if sum(self.weights) <= 0:
            self.weights = [1.0 for _ in sources]

    def choose(self) -> DataSource:
        return random.choices(self.sources, weights=self.weights, k=1)[0]


class CausalLMDataset(Dataset):
    """
    Streaming-friendly causal LM dataset with optional packing.
    """

    def __init__(
        self,
        tokenizer,
        sources: List[DataSource],
        mixture: DataMixtureConfig,
        pack_sequences: bool | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sources = sources
        self.mixture = mixture
        self.pack_sequences = pack_sequences if pack_sequences is not None else getattr(
            mixture, "pack_sequences", True
        )
        self.samples_per_epoch = getattr(mixture, "samples_per_epoch", 1024)
        self.max_seq_len = mixture.max_sequence_length
        self.add_bos = getattr(mixture, "add_bos", True)
        self.add_eos = getattr(mixture, "add_eos", True)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", getattr(tokenizer, "eos_id", 2))
        self.bos_token_id = getattr(tokenizer, "bos_token_id", getattr(tokenizer, "bos_id", 1))

        self.mixture_policy = MixturePolicy(sources)
        self._iterators: Dict[str, Iterator[List[int]]] = {
            s.name: iter(self._stream_source(s)) for s in sources
        }
        self._pack_buffers: Dict[str, List[int]] = {s.name: [] for s in sources}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # idx is ignored because sampling is stochastic per mixture weights
        source = self.mixture_policy.choose()
        iterator = self._iterators[source.name]
        for _ in range(2):  # retry once after refreshing iterator
            try:
                seq = next(iterator)
                break
            except StopIteration:
                self._iterators[source.name] = iter(self._stream_source(source))
                iterator = self._iterators[source.name]
        else:
            # If no data is available for this source, fall back to a synthetic sample
            seq = self._make_dummy_sequence()
        input_ids = torch.tensor(seq, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        padded_ids, padded_mask = self._pad(input_ids, attention_mask)
        return padded_ids, padded_mask

    def _stream_source(self, source: DataSource) -> Iterator[List[int]]:
        for text in iter_normalized(source.stream_documents()):
            token_ids = self.tokenizer(
                text,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
            if not isinstance(token_ids, list):
                token_ids = list(token_ids)
            if self.pack_sequences:
                yield from self._pack_tokens(source.name, token_ids)
            else:
                yield token_ids[: self.max_seq_len]
        # Flush remaining packed tokens for this source
        if self.pack_sequences:
            buffer = self._pack_buffers[source.name]
            if buffer:
                yield buffer[: self.max_seq_len]
                buffer.clear()

    def _pack_tokens(self, source_name: str, token_sequence: List[int]) -> Iterator[List[int]]:
        """
        Pack multiple short sequences into one window to improve efficiency.
        """
        buffer = self._pack_buffers[source_name]
        for tok in token_sequence:
            buffer.append(tok)
            if len(buffer) == self.max_seq_len:
                yield buffer.copy()
                buffer.clear()
        # Ensure EOS boundary between packed samples
        if self.add_eos and buffer and buffer[-1] != self.eos_token_id:
            if len(buffer) < self.max_seq_len:
                buffer.append(self.eos_token_id)
            else:
                yield buffer.copy()
                buffer.clear()
                buffer.append(self.eos_token_id)
        if len(buffer) >= self.max_seq_len:
            yield buffer[: self.max_seq_len]
            del buffer[: self.max_seq_len]

    def _pad(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_len = self.max_seq_len - input_ids.shape[0]
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
        return input_ids[: self.max_seq_len], attention_mask[: self.max_seq_len]

    def _make_dummy_sequence(self) -> List[int]:
        """
        Fallback synthetic sequence to avoid runtime failure when sources are empty/missing.
        """
        # Reserve space for BOS/EOS if enabled
        usable_len = self.max_seq_len
        tokens: List[int] = []
        if self.add_bos:
            tokens.append(self.bos_token_id)
            usable_len -= 1
        if self.add_eos:
            usable_len -= 1

        # Fill with random ids
        vocab = getattr(self.tokenizer, "vocab_size", 32000)
        if usable_len > 0:
            random_ids = torch.randint(0, vocab, (usable_len,)).tolist()
            tokens.extend(random_ids)

        # Append EOS if requested
        if self.add_eos:
            tokens.append(self.eos_token_id)

        return tokens[: self.max_seq_len]
