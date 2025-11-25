from __future__ import annotations

import random

from havoc_core.config import DataMixtureConfig
from havoc_data.dataset import CausalLMDataset
from havoc_data.preprocess import iter_normalized, normalize_symbols, tag_dsl
from havoc_data.sources import InMemorySource


class TinyTokenizer:
    """Whitespace tokenizer with fixed IDs for tests."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    token_map = {
        "A": 11,
        "B": 22,
        "hello": 33,
        "world": 44,
    }

    def __call__(self, text: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.token_map.get(tok, 99) for tok in text.split()]
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
        return tokens


def test_mixture_weighting_prefers_higher_weight():
    random.seed(0)
    tokenizer = TinyTokenizer()
    mixture = DataMixtureConfig(max_sequence_length=4, samples_per_epoch=200, add_bos=True, add_eos=True)

    source_a = InMemorySource("a", ["A"], weight=5.0)
    source_b = InMemorySource("b", ["B"], weight=1.0)
    ds = CausalLMDataset(tokenizer, [source_a, source_b], mixture)

    counts = {"a": 0, "b": 0}
    for _ in range(len(ds)):
        input_ids, _ = ds[_]
        if tokenizer.token_map["A"] in input_ids.tolist():
            counts["a"] += 1
        if tokenizer.token_map["B"] in input_ids.tolist():
            counts["b"] += 1
    assert counts["a"] > counts["b"] * 2  # should heavily favor higher weight


def test_bos_eos_and_padding():
    tokenizer = TinyTokenizer()
    mixture = DataMixtureConfig(
        max_sequence_length=6,
        samples_per_epoch=1,
        add_bos=True,
        add_eos=True,
        pack_sequences=False,
    )
    source = InMemorySource("single", ["hello world"], weight=1.0)
    ds = CausalLMDataset(tokenizer, [source], mixture)
    input_ids, attention_mask = ds[0]

    assert input_ids[0].item() == tokenizer.bos_token_id
    assert input_ids[3].item() == tokenizer.eos_token_id
    assert input_ids[-1].item() == tokenizer.pad_token_id  # padded
    assert attention_mask.sum().item() == 4  # bos + 2 tokens + eos


def test_dsl_tagging_and_normalization():
    line = '{"MATH": {"expression": "x**2", "variables": {"x": 2}}}'
    tagged = tag_dsl(line)
    assert tagged.startswith("<DSL_BEGIN>")
    normalized_lines = list(iter_normalized([line]))
    assert normalized_lines and "<DSL_BEGIN>" in normalized_lines[0]

    sym_line = "µ = 2 * σ"
    assert "mu" in normalize_symbols(sym_line)
    assert "sigma" in normalize_symbols(sym_line)
