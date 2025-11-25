from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from havoc_core.config import DataMixtureConfig
from havoc_core.tokenizer.tokenizer import HavocTokenizer
from havoc_data.dataset import CausalLMDataset
from havoc_data.sources import InMemorySource, build_source_from_dict, load_sources


class DummyTokenizer:
    """
    Lightweight whitespace tokenizer used when no SentencePiece model is provided.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __call__(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = text.split()
        ids = [hash(tok) % self.vocab_size for tok in tokens]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids


def _load_tokenizer(path: Optional[str]) -> object:
    if path and Path(path).exists():
        return HavocTokenizer(path)
    return DummyTokenizer()


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_data_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config_path: str) -> Path:
    cfg = load_data_config(config_path)
    mixture = DataMixtureConfig(**cfg.get("mixture", {}))
    sources_def = cfg.get("sources", [])
    sources = load_sources(sources_def)
    if not sources:
        # Provide a tiny synthetic corpus so the pipeline still runs
        sources = [
            InMemorySource(
                name="synthetic",
                items=[
                    "General text sample for sanity check.",
                    "<DSL_BEGIN> RUN_TTEST alpha=0.05 <DSL_END>",
                    "Engineering: stress = force / area, sigma = F/A",
                ],
            )
        ]
    tokenizer_path = cfg.get("tokenizer_path")
    tokenizer = _load_tokenizer(tokenizer_path)

    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=sources,
        mixture=mixture,
        pack_sequences=getattr(mixture, "pack_sequences", True),
    )

    output_path = Path(cfg.get("output_path", "artifacts/data/train.jsonl"))
    _ensure_output_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as handle:
        for idx in range(len(dataset)):
            input_ids, attention_mask = dataset[idx]
            record = {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
            }
            handle.write(json.dumps(record) + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tokenized dataset from YAML config.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    args = parser.parse_args()

    output = build_dataset(args.config)
    print(f"Dataset built at {output}")


if __name__ == "__main__":
    main()
