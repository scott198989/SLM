from __future__ import annotations

import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def clean_line(line: str) -> str:
    cleaned = normalize_whitespace(line)
    cleaned = cleaned.replace("\u00b5", "mu").replace("\u03c3", "sigma")
    return cleaned


def iter_normalized(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        normalized = clean_line(line)
        if normalized:
            yield normalized
