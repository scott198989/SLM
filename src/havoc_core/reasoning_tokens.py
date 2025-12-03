"""
Reasoning Token System for HAVOC-7B

Provides utilities for managing reasoning tokens in chain-of-thought generation.
These tokens make the model's reasoning process explicit and visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import re


class ReasoningTokenType(Enum):
    """Types of reasoning tokens"""
    REASON_START = "<reason>"
    REASON_END = "</reason>"
    TOOL_START = "<tool>"
    TOOL_END = "</tool>"
    ADVOCATE = "<advocate>"
    ADVOCATE_END = "</advocate>"
    ATTACK = "<attack>"
    ATTACK_END = "</attack>"
    PRAGMATIST = "<pragmatist>"
    PRAGMATIST_END = "</pragmatist>"


# Standard token IDs (must match tokenizer configuration)
REASONING_TOKEN_IDS = {
    ReasoningTokenType.REASON_START: 10,
    ReasoningTokenType.REASON_END: 11,
    ReasoningTokenType.TOOL_START: 12,
    ReasoningTokenType.TOOL_END: 13,
    ReasoningTokenType.ADVOCATE: 14,
    ReasoningTokenType.ADVOCATE_END: 15,
    ReasoningTokenType.ATTACK: 16,
    ReasoningTokenType.ATTACK_END: 17,
    ReasoningTokenType.PRAGMATIST: 18,
    ReasoningTokenType.PRAGMATIST_END: 19,
}


@dataclass
class ReasoningSegment:
    """A segment of reasoning in the output"""
    token_type: ReasoningTokenType
    content: str
    start_pos: int
    end_pos: int


class ReasoningTokenParser:
    """Parse and extract reasoning segments from generated text"""

    def __init__(self):
        self.token_patterns = {
            "reason": (r"<reason>(.*?)</reason>", ReasoningTokenType.REASON_START),
            "tool": (r"<tool>(.*?)</tool>", ReasoningTokenType.TOOL_START),
            "advocate": (r"<advocate>(.*?)</advocate>", ReasoningTokenType.ADVOCATE),
            "attack": (r"<attack>(.*?)</attack>", ReasoningTokenType.ATTACK),
            "pragmatist": (r"<pragmatist>(.*?)</pragmatist>", ReasoningTokenType.PRAGMATIST),
        }

    def parse(self, text: str) -> List[ReasoningSegment]:
        """
        Parse text and extract all reasoning segments.

        Args:
            text: Generated text containing reasoning tokens

        Returns:
            List of ReasoningSegment objects
        """
        segments = []

        for name, (pattern, token_type) in self.token_patterns.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                segment = ReasoningSegment(
                    token_type=token_type,
                    content=match.group(1).strip(),
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                segments.append(segment)

        # Sort by position
        segments.sort(key=lambda s: s.start_pos)
        return segments

    def extract_reasoning(self, text: str) -> str:
        """Extract only <reason>...</reason> content"""
        segments = self.parse(text)
        reason_segments = [
            s.content for s in segments
            if s.token_type == ReasoningTokenType.REASON_START
        ]
        return "\n\n".join(reason_segments)

    def extract_tool_calls(self, text: str) -> List[str]:
        """Extract all <tool>...</tool> content"""
        segments = self.parse(text)
        return [
            s.content for s in segments
            if s.token_type == ReasoningTokenType.TOOL_START
        ]

    def extract_adversarial_reasoning(self, text: str) -> Tuple[str, str, str]:
        """
        Extract adversarial reasoning segments.

        Returns:
            Tuple of (advocate, attack, pragmatist) content
        """
        segments = self.parse(text)

        advocate = ""
        attack = ""
        pragmatist = ""

        for segment in segments:
            if segment.token_type == ReasoningTokenType.ADVOCATE:
                advocate = segment.content
            elif segment.token_type == ReasoningTokenType.ATTACK:
                attack = segment.content
            elif segment.token_type == ReasoningTokenType.PRAGMATIST:
                pragmatist = segment.content

        return advocate, attack, pragmatist

    def strip_reasoning_tokens(self, text: str) -> str:
        """
        Remove all reasoning tokens from text, leaving only final answer.

        NOTE: This is OPTIONAL - by default reasoning tokens should be visible.
        """
        result = text

        # Remove all reasoning token pairs
        for pattern, _ in self.token_patterns.values():
            result = re.sub(pattern, "", result, flags=re.DOTALL)

        # Clean up extra whitespace
        result = re.sub(r'\n\n+', '\n\n', result)
        return result.strip()


class ReasoningTokenFormatter:
    """Format reasoning for output with reasoning tokens"""

    @staticmethod
    def format_reason(content: str) -> str:
        """Wrap content in <reason>...</reason>"""
        return f"<reason>\n{content}\n</reason>"

    @staticmethod
    def format_tool_call(tool_name: str, args: dict) -> str:
        """Format a tool call with <tool>...</tool>"""
        import json
        tool_json = json.dumps({"tool": tool_name, "args": args}, indent=2)
        return f"<tool>\n{tool_json}\n</tool>"

    @staticmethod
    def format_advocate(content: str) -> str:
        """Format advocate argument"""
        return f"<advocate>\n{content}\n</advocate>"

    @staticmethod
    def format_attack(content: str) -> str:
        """Format attack argument"""
        return f"<attack>\n{content}\n</attack>"

    @staticmethod
    def format_pragmatist(content: str) -> str:
        """Format pragmatist synthesis"""
        return f"<pragmatist>\n{content}\n</pragmatist>"

    @staticmethod
    def format_full_reasoning(
        reason: str,
        tool_calls: List[Tuple[str, dict]] = None,
        advocate: str = None,
        attack: str = None,
        pragmatist: str = None,
        final_answer: str = ""
    ) -> str:
        """
        Format complete reasoning chain.

        Args:
            reason: Chain-of-thought reasoning
            tool_calls: List of (tool_name, args) tuples
            advocate: Advocate's argument (optional)
            attack: Attack's argument (optional)
            pragmatist: Pragmatist's synthesis (optional)
            final_answer: Final answer

        Returns:
            Formatted string with all reasoning tokens
        """
        parts = []

        # Add reasoning
        if reason:
            parts.append(ReasoningTokenFormatter.format_reason(reason))

        # Add tool calls
        if tool_calls:
            for tool_name, args in tool_calls:
                parts.append(ReasoningTokenFormatter.format_tool_call(tool_name, args))

        # Add adversarial reasoning
        if advocate:
            parts.append(ReasoningTokenFormatter.format_advocate(advocate))
        if attack:
            parts.append(ReasoningTokenFormatter.format_attack(attack))
        if pragmatist:
            parts.append(ReasoningTokenFormatter.format_pragmatist(pragmatist))

        # Add final answer
        if final_answer:
            parts.append(final_answer)

        return "\n\n".join(parts)


def validate_reasoning_tokens(text: str) -> Tuple[bool, List[str]]:
    """
    Validate that reasoning tokens are properly balanced.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check for balanced <reason>...</reason>
    reason_starts = text.count("<reason>")
    reason_ends = text.count("</reason>")
    if reason_starts != reason_ends:
        errors.append(f"Unbalanced <reason> tags: {reason_starts} starts, {reason_ends} ends")

    # Check for balanced <tool>...</tool>
    tool_starts = text.count("<tool>")
    tool_ends = text.count("</tool>")
    if tool_starts != tool_ends:
        errors.append(f"Unbalanced <tool> tags: {tool_starts} starts, {tool_ends} ends")

    # Check for balanced <advocate>...</advocate>
    advocate_starts = text.count("<advocate>")
    advocate_ends = text.count("</advocate>")
    if advocate_starts != advocate_ends:
        errors.append(f"Unbalanced <advocate> tags: {advocate_starts} starts, {advocate_ends} ends")

    # Check for balanced <attack>...</attack>
    attack_starts = text.count("<attack>")
    attack_ends = text.count("</attack>")
    if attack_starts != attack_ends:
        errors.append(f"Unbalanced <attack> tags: {attack_starts} starts, {attack_ends} ends")

    # Check for balanced <pragmatist>...</pragmatist>
    pragmatist_starts = text.count("<pragmatist>")
    pragmatist_ends = text.count("</pragmatist>")
    if pragmatist_starts != pragmatist_ends:
        errors.append(f"Unbalanced <pragmatist> tags: {pragmatist_starts} starts, {pragmatist_ends} ends")

    return len(errors) == 0, errors


# Example usage
if __name__ == "__main__":
    # Example: Parse reasoning from generated text
    example_text = """
    <reason>
    Task: Compare two groups using t-test
    Subgoal 1: Extract data from prompt
    Subgoal 2: Run t-test tool
    Subgoal 3: Interpret results
    </reason>

    <tool>
    {"tool": "python_math", "args": {"operation": "t_test", "group1": [1,2,3], "group2": [2,3,4]}}
    </tool>

    <advocate>
    The t-test shows a significant difference (p < 0.05), so we should reject the null hypothesis.
    </advocate>

    <attack>
    However, the sample sizes are very small (n=3), which reduces statistical power.
    </attack>

    <pragmatist>
    While statistically significant, the small sample size warrants caution. Recommend collecting more data.
    </pragmatist>

    Final answer: The groups show a significant difference (p=0.048), but small sample size (n=3) limits confidence.
    """

    parser = ReasoningTokenParser()

    # Parse all segments
    segments = parser.parse(example_text)
    print(f"Found {len(segments)} reasoning segments:")
    for seg in segments:
        print(f"  - {seg.token_type.value}: {seg.content[:50]}...")

    # Extract specific components
    print("\nReasoning:")
    print(parser.extract_reasoning(example_text))

    print("\nTool calls:")
    for tool_call in parser.extract_tool_calls(example_text):
        print(f"  {tool_call}")

    print("\nAdversarial reasoning:")
    advocate, attack, pragmatist = parser.extract_adversarial_reasoning(example_text)
    print(f"  Advocate: {advocate[:50]}...")
    print(f"  Attack: {attack[:50]}...")
    print(f"  Pragmatist: {pragmatist[:50]}...")

    # Validate
    is_valid, errors = validate_reasoning_tokens(example_text)
    print(f"\nValidation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
