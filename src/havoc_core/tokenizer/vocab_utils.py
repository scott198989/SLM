from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

DOMAIN_TOKENS = [
    "ANOVA",
    "p-value",
    "control_chart",
    "Box-Behnken",
    "Taguchi",
    "Cpk",
    "Cp",
    "Ppk",
    "CHECK_SPC",
    "EVAL_DOE",
    "RUN_TTEST",
    "DEFINE_OPERATOR",
    "FACTOR",
    "RESPONSE",
    "ALPHA_0_05",
    "sigma",
    "mu",
    "integral",
    "derivative",
]


@dataclass
class TokenizerMetadata:
    vocab_size: int
    special_tokens: List[str]
    domain_tokens: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "domain_tokens": self.domain_tokens,
        }


def register_special_tokens(base: List[str]) -> List[str]:
    merged = base + [tok for tok in DOMAIN_TOKENS if tok not in base]
    return merged


def sample_domain_strings() -> List[str]:
    return [
        "Perform an ANOVA with alpha=0.05 on the Box-Behnken design and report Cpk/Cp.",
        "RUN_TTEST between treatment A and B with subgroup size 5.",
        "CHECK_SPC XR chart for torque and evaluate Ppk.",
    ]
