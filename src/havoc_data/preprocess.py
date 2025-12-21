from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple


WHITESPACE_RE = re.compile(r"\s+")
DSL_BLOCK_RE = re.compile(r"```dsl\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
REASONING_TRACE_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Math and engineering symbols to normalize
SYMBOL_REPLACEMENTS = {
    "\u00b5": "mu",  # micro
    "\u03bc": "mu",  # μ
    "\u03c3": "sigma",  # σ
    "\u03a3": "Sigma",  # Σ
    "\u03c0": "pi",  # π
    "\u03b1": "alpha",  # α
    "\u03b2": "beta",  # β
    "\u03b3": "gamma",  # γ
    "\u03b4": "delta",  # δ
    "\u0394": "Delta",  # Δ
    "\u03b5": "epsilon",  # ε
    "\u03b8": "theta",  # θ
    "\u03bb": "lambda",  # λ
    "\u03c1": "rho",  # ρ
    "\u03c4": "tau",  # τ
    "\u03c6": "phi",  # φ
    "\u03c7": "chi",  # χ
    "\u03c9": "omega",  # ω
    "\u00b1": "+/-",  # ±
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u2260": "!=",  # ≠
    "\u221a": "sqrt",  # √
    "\u222b": "integral",  # ∫
    "\u2211": "sum",  # ∑
    "\u220f": "product",  # ∏
    "\u221e": "infinity",  # ∞
    "\u2192": "->",  # →
    "\u2190": "<-",  # ←
    "\u00b0": "deg",  # °
}


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace to single spaces."""
    return WHITESPACE_RE.sub(" ", text.strip())


def normalize_symbols(text: str) -> str:
    """Replace Unicode math/engineering symbols with ASCII equivalents."""
    result = text
    for symbol, replacement in SYMBOL_REPLACEMENTS.items():
        result = result.replace(symbol, replacement)
    return result


def extract_dsl_blocks(text: str) -> Tuple[str, list[str]]:
    """Extract DSL code blocks and replace with tagged versions.

    Returns:
        Tuple of (modified_text, list_of_dsl_blocks)
    """
    dsl_blocks = []

    def replace_dsl(match):
        dsl_code = match.group(1).strip()
        dsl_blocks.append(dsl_code)
        return f"<DSL_BEGIN>\n{dsl_code}\n<DSL_END>"

    modified_text = DSL_BLOCK_RE.sub(replace_dsl, text)
    return modified_text, dsl_blocks


def annotate_reasoning_traces(text: str) -> str:
    """Annotate reasoning traces with special markers.

    Converts <think>...</think> blocks to structured reasoning traces.
    """
    def replace_reasoning(match):
        reasoning = match.group(1).strip()
        return f"<REASONING_BEGIN>\n{reasoning}\n<REASONING_END>"

    return REASONING_TRACE_RE.sub(replace_reasoning, text)


def is_malformed(line: str, min_length: int = 10, max_length: int = 10000) -> bool:
    """Check if a line is malformed and should be rejected.

    Args:
        line: The line to check
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length

    Returns:
        True if line should be rejected, False otherwise
    """
    # Check length
    if len(line) < min_length or len(line) > max_length:
        return True

    # Check for excessive special characters (likely corrupted)
    special_char_ratio = sum(1 for c in line if not c.isalnum() and not c.isspace()) / len(line)
    if special_char_ratio > 0.5:
        return True

    # Check for valid content (must have some alphanumeric)
    if not any(c.isalnum() for c in line):
        return True

    return False


def clean_line(line: str) -> str:
    """Clean and normalize a single line of text."""
    cleaned = normalize_whitespace(line)
    cleaned = normalize_symbols(cleaned)
    return cleaned


def preprocess_text(text: str, extract_dsl: bool = True, annotate_reasoning: bool = True) -> str:
    """Full preprocessing pipeline for text.

    Args:
        text: Raw text to preprocess
        extract_dsl: Whether to extract and tag DSL blocks
        annotate_reasoning: Whether to annotate reasoning traces

    Returns:
        Preprocessed text
    """
    result = text

    # Extract DSL blocks first
    if extract_dsl:
        result, _ = extract_dsl_blocks(result)

    # Annotate reasoning traces
    if annotate_reasoning:
        result = annotate_reasoning_traces(result)

    # Normalize symbols
    result = normalize_symbols(result)

    return result


def iter_normalized(
    lines: Iterable[str],
    reject_malformed: bool = True,
    extract_dsl: bool = False,
    annotate_reasoning: bool = False,
) -> Iterable[str]:
    """Iterate over normalized lines, optionally rejecting malformed ones.

    Args:
        lines: Input lines
        reject_malformed: Whether to reject malformed lines
        extract_dsl: Whether to extract DSL blocks
        annotate_reasoning: Whether to annotate reasoning traces

    Yields:
        Cleaned, normalized lines
    """
    for line in lines:
        # Apply preprocessing
        if extract_dsl or annotate_reasoning:
            line = preprocess_text(line, extract_dsl, annotate_reasoning)

        # Clean the line
        normalized = clean_line(line)

        # Check if malformed
        if normalized and (not reject_malformed or not is_malformed(normalized)):
            yield normalized


def normalize_text(text: str, extract_dsl: bool = True, annotate_reasoning: bool = True) -> str:
    """Normalize a complete text document.

    This is the main entry point for text preprocessing.

    Args:
        text: Raw text
        extract_dsl: Whether to extract and tag DSL blocks
        annotate_reasoning: Whether to annotate reasoning traces

    Returns:
        Normalized text
    """
    return preprocess_text(text, extract_dsl, annotate_reasoning)
