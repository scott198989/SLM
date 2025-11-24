from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm

from havoc_core.config import TokenizerTrainingConfig
from havoc_core.tokenizer.vocab_utils import TokenizerMetadata, register_special_tokens, sample_domain_strings


def normalize_text(line: str) -> str:
    return " ".join(line.strip().split())


def iter_corpus(paths: List[str], normalize: bool = True) -> Iterable[str]:
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for file in p.rglob("*.txt"):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        yield normalize_text(line) if normalize else line.strip()
        else:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    yield normalize_text(line) if normalize else line.strip()


def train_tokenizer(config: TokenizerTrainingConfig) -> TokenizerMetadata:
    os.makedirs(config.output_dir, exist_ok=True)
    special_tokens = register_special_tokens(config.special_tokens)
    training_corpus = list(iter_corpus(config.input_files, normalize=config.normalize_text))
    training_corpus.extend(sample_domain_strings())

    model_prefix = Path(config.output_dir) / "havoc_tokenizer"
    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(training_corpus),
        model_prefix=str(model_prefix),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        max_sentence_length=config.max_sentence_length,
        input_sentence_size=len(training_corpus),
        shuffle_input_sentence=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=special_tokens,
    )

    return TokenizerMetadata(
        vocab_size=config.vocab_size,
        special_tokens=special_tokens,
        domain_tokens=[tok for tok in special_tokens if tok not in config.special_tokens],
    )


if __name__ == "__main__":
    cfg = TokenizerTrainingConfig()
    metadata = train_tokenizer(cfg)
    print("Tokenizer artifacts stored in", cfg.output_dir)
    print(metadata.as_dict())
