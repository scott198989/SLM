from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset

from havoc_core.config import DataMixtureConfig
from havoc_data.preprocess import iter_normalized
from havoc_data.sources import DataSource


@dataclass
class SequenceExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class CausalLMDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        sources: List[DataSource],
        mixture: DataMixtureConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.sources = sources
        self.mixture = mixture
        self.samples = list(self._build_samples())

    def _build_samples(self) -> Iterable[SequenceExample]:
        for source in self.sources:
            for file in source.files():
                text = Path(file).read_text(encoding="utf-8")
                for paragraph in iter_normalized(text.splitlines()):
                    tokens = self.tokenizer(paragraph)
                    input_ids = torch.tensor(tokens[: self.mixture.max_sequence_length])
                    attn_mask = torch.ones_like(input_ids)
                    yield SequenceExample(input_ids=input_ids, attention_mask=attn_mask)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.samples[idx]
        padded_ids = self._pad(ex.input_ids)
        padded_mask = self._pad(ex.attention_mask)
        return padded_ids, padded_mask

    def _pad(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_len = self.mixture.max_sequence_length - tensor.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)
        return tensor[: self.mixture.max_sequence_length]


class MixturePolicy:
    def __init__(self, mixture: DataMixtureConfig):
        self.mixture = mixture

    def choose_source(self, sources: List[DataSource]) -> DataSource:
        weights = [s.weight for s in sources]
        return random.choices(sources, weights=weights, k=1)[0]
