from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# SRS (Scott Reasoning Stack) stage markers
SRS_TOKENS = [
    "<SRS_MODE>",
    "<SRS_GROUND>",
    "<SRS_PLAN>",
    "<SRS_EXECUTE>",
    "<SRS_ARGUE>",
    "<SRS_ARBITER>",
    "<SRS_AUDIT>",
    "<SRS_ANSWER>",
]

# DSL (Domain-Specific Language) markers
DSL_TOKENS = [
    "<DSL_BEGIN>",
    "<DSL_END>",
]

# Tool invocation markers
TOOL_TOKENS = [
    "<TOOL_MATH>",
    "<TOOL_STATS>",
]

# Engineering symbol boundaries
ENGINEERING_TOKENS = [
    "<ENG_SYMBOL_START>",
    "<ENG_SYMBOL_END>",
]

# Math and engineering symbols - these should be preserved in tokenization
MATH_SYMBOLS = [
    "∑",  # Sum
    "∏",  # Product
    "∫",  # Integral
    "∂",  # Partial derivative
    "∇",  # Nabla/gradient
    "√",  # Square root
    "≈",  # Approximately equal
    "≠",  # Not equal
    "≤",  # Less than or equal
    "≥",  # Greater than or equal
    "±",  # Plus-minus
    "×",  # Multiplication
    "÷",  # Division
    "∞",  # Infinity
]

# Greek letters commonly used in math/stats/engineering
GREEK_LETTERS = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ",  # Lowercase
    "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π",
    "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
    "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ",  # Uppercase
    "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π",
    "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
]

# Engineering units and common expressions
ENGINEERING_UNITS = [
    "psi",
    "MPa",
    "GPa",
    "kPa",
    "N/m",
    "N/cm",
    "N/mm",
    "kg/m³",
    "m/s",
    "m/s²",
    "kW",
    "MW",
    "kWh",
    "°C",
    "°F",
    "K",  # Kelvin
    "Hz",
    "kHz",
    "MHz",
    "GHz",
]

# DOE/SPC domain-specific tokens
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
    "X-bar",
    "R-chart",
    "UCL",  # Upper Control Limit
    "LCL",  # Lower Control Limit
    "CL",   # Center Line
    "factorial",
    "fractional_factorial",
    "central_composite",
    "Plackett-Burman",
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


def get_all_special_tokens() -> List[str]:
    """Collect all special tokens from all categories."""
    all_tokens = []

    # Add SRS tokens
    all_tokens.extend(SRS_TOKENS)

    # Add DSL tokens
    all_tokens.extend(DSL_TOKENS)

    # Add tool tokens
    all_tokens.extend(TOOL_TOKENS)

    # Add engineering boundary tokens
    all_tokens.extend(ENGINEERING_TOKENS)

    # Add math symbols
    all_tokens.extend(MATH_SYMBOLS)

    # Add Greek letters
    all_tokens.extend(GREEK_LETTERS)

    # Add engineering units
    all_tokens.extend(ENGINEERING_UNITS)

    # Add domain tokens
    all_tokens.extend(DOMAIN_TOKENS)

    return all_tokens


def register_special_tokens(base: List[str]) -> List[str]:
    """Merge base tokens with all domain-specific special tokens."""
    all_special = get_all_special_tokens()
    merged = base + [tok for tok in all_special if tok not in base]
    return merged


def sample_domain_strings() -> List[str]:
    """Generate sample strings that demonstrate usage of special tokens and domain vocabulary."""
    return [
        # SRS reasoning examples
        "<SRS_MODE> Identify problem type: statistical hypothesis testing <SRS_GROUND>",
        "<SRS_PLAN> Design experiment → collect data → analyze → conclude <SRS_EXECUTE>",
        "<SRS_ARGUE> H0: μ₁ = μ₂ vs H1: μ₁ ≠ μ₂ <SRS_ARBITER>",
        "<SRS_AUDIT> Assumptions: normality ✓, equal variance ✓ <SRS_ANSWER>",

        # DSL examples
        "<DSL_BEGIN> CHECK_SPC X-bar chart, n=5, UCL=100, LCL=80 <DSL_END>",
        "<DSL_BEGIN> EVAL_DOE factorial 2³, RESPONSE=yield, FACTOR=temp,pressure,time <DSL_END>",
        "<DSL_BEGIN> RUN_TTEST alpha=0.05, two-tailed <DSL_END>",

        # Tool invocation examples
        "<TOOL_MATH> ∫₀¹ x² dx = 1/3 </TOOL_MATH>",
        "<TOOL_STATS> ANOVA F(2,27)=5.43, p-value=0.011 </TOOL_STATS>",

        # Engineering symbols
        "<ENG_SYMBOL_START> σ = 250 MPa, ε = 0.02 <ENG_SYMBOL_END>",
        "<ENG_SYMBOL_START> P = 150 psi, A = 10 N/cm² <ENG_SYMBOL_END>",

        # Math expressions
        "The stress is σ = F/A where F is force and A is area.",
        "Calculate mean μ and standard deviation σ from the sample.",
        "The gradient ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z) at point P.",
        "Sum: ∑ᵢ₌₁ⁿ xᵢ = 100, Product: ∏ᵢ₌₁ⁿ xᵢ = 1000",

        # Statistical analysis
        "Perform an ANOVA with alpha=0.05 on the Box-Behnken design and report Cpk/Cp.",
        "RUN_TTEST between treatment A and B with subgroup size 5.",
        "CHECK_SPC X-bar/R-chart for torque and evaluate Ppk.",
        "Taguchi L₉ orthogonal array with 3 factors at 3 levels each.",
        "Central composite design with 5 center points, α = 1.414.",
        "Plackett-Burman screening design for 7 factors in 8 runs.",

        # Engineering calculations
        "Material properties: E = 200 GPa, ν = 0.3, ρ = 7850 kg/m³",
        "Operating conditions: T = 150°C, P = 5 MPa, flow rate = 2.5 m³/s",
        "Power calculation: P = VI = 230V × 15A = 3.45 kW",
        "Frequency response: f₀ = 1/(2π√(LC)) ≈ 1.59 kHz",

        # Mixed content
        "The process capability Cpk = min((USL-μ)/(3σ), (μ-LSL)/(3σ)) ≥ 1.33",
        "Confidence interval: μ ± t₀.₀₂₅ × (s/√n) = 50 ± 2.05",
        "Hypothesis test: if p-value < α = 0.05, reject H₀",
    ]


def get_normalization_rules() -> Dict[str, str]:
    """
    Returns a dictionary of normalization rules for mathematical and engineering text.
    Maps unicode characters to their preferred representations.
    """
    return {
        # Superscripts to caret notation
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "⁰": "^0",
        "¹": "^1",

        # Subscripts to underscore notation
        "₀": "_0",
        "₁": "_1",
        "₂": "_2",
        "₃": "_3",
        "₄": "_4",
        "₅": "_5",
        "₆": "_6",
        "₇": "_7",
        "₈": "_8",
        "₉": "_9",

        # Alternative representations to standard
        "−": "-",  # Minus sign to hyphen-minus
        "–": "-",  # En dash to hyphen-minus
        "—": "-",  # Em dash to hyphen-minus
        "'": "'",  # Right single quotation mark to apostrophe
        "'": "'",  # Left single quotation mark to apostrophe
        """: '"',  # Left double quotation mark
        """: '"',  # Right double quotation mark
        "…": "...",  # Horizontal ellipsis

        # Preserve these special math symbols (no normalization)
        # These are handled by adding them to the vocabulary
    }


def get_char_normalization_map() -> Dict[str, str]:
    """
    Returns character-level normalization for safe ASCII conversion.
    Only normalizes where it doesn't change mathematical meaning.
    """
    rules = get_normalization_rules()

    # Add common fraction characters (optional, depending on desired behavior)
    rules.update({
        "½": "1/2",
        "⅓": "1/3",
        "⅔": "2/3",
        "¼": "1/4",
        "¾": "3/4",
        "⅛": "1/8",
        "⅜": "3/8",
        "⅝": "5/8",
        "⅞": "7/8",
    })

    return rules
