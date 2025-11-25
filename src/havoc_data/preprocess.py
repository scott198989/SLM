from __future__ import annotations

import json
import re
from typing import Iterable, Iterator, Optional

from havoc_core.tokenizer.vocab_utils import get_char_normalization_map

EXTRA_SYMBOL_MAP = {
    "µ": "mu",
    "μ": "mu",
    "σ": "sigma",
    "Σ": "Sigma",
    "±": "+/-",
    "∓": "+/-",
    "×": "x",
    "·": "*",
}

# Pre-compiled regexes for speed
WHITESPACE_RE = re.compile(r"\s+")
DSL_KEYWORDS_RE = re.compile(
    r"(CHECK_SPC|EVAL_DOE|RUN_TTEST|STAT_TEST|DOE|SPC|MATH)", re.IGNORECASE
)
REASONING_RE = re.compile(r"(reasoning|trace|chain of thought):", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the ends."""
    return WHITESPACE_RE.sub(" ", text.strip())


def normalize_symbols(text: str) -> str:
    """Normalize math/engineering symbols to ASCII-friendly tokens."""
    rules = get_char_normalization_map()
    normalized = []
    for ch in text:
        if ch in EXTRA_SYMBOL_MAP:
            normalized.append(EXTRA_SYMBOL_MAP[ch])
        elif ch in rules:
            normalized.append(rules[ch])
        else:
            normalized.append(ch)
    return "".join(normalized)


def tag_dsl(text: str) -> str:
    """
    Detect DSL-like payloads and wrap them with DSL markers.

    This catches JSON/YAML snippets that contain DSL keys or shorthand commands.
    """
    if "<DSL_BEGIN>" in text and "<DSL_END>" in text:
        return text

    looks_like_json = text.strip().startswith("{") and text.strip().endswith("}")
    if looks_like_json:
        try:
            data = json.loads(text)
            if any(key in data for key in ("MATH", "STAT_TEST", "DOE", "SPC")):
                return f"<DSL_BEGIN> {text} <DSL_END>"
        except Exception:
            pass

    if DSL_KEYWORDS_RE.search(text):
        return f"<DSL_BEGIN> {text} <DSL_END>"
    return text


def annotate_reasoning(text: str) -> str:
    """
    Add SRS reasoning markers around explicit reasoning traces when present.
    """
    match = REASONING_RE.search(text)
    if not match:
        return text

    idx = match.start()
    prefix, rest = text[:idx], text[idx:]
    return f"{prefix}<SRS_PLAN> {rest} <SRS_ANSWER>"


def is_malformed(text: str) -> bool:
    """
    Simple heuristic for malformed lines:
    - empty after normalization
    - unmatched DSL boundary markers
    """
    if not text:
        return True
    if text.count("<DSL_BEGIN>") != text.count("<DSL_END>"):
        return True
    return False


def clean_line(line: str) -> Optional[str]:
    """Full preprocessing pipeline for a single line."""
    cleaned = normalize_whitespace(line)
    cleaned = normalize_symbols(cleaned)
    cleaned = tag_dsl(cleaned)
    cleaned = annotate_reasoning(cleaned)
    cleaned = cleaned.strip()
    if is_malformed(cleaned):
        return None
    return cleaned


def iter_normalized(lines: Iterable[str]) -> Iterator[str]:
    """Yield normalized, well-formed lines."""
    for line in lines:
        normalized = clean_line(line)
        if normalized:
            yield normalized


def normalize_text(text: str) -> str:
    """Convenience wrapper used by callers that expect a simple function."""
    return clean_line(text) or ""
