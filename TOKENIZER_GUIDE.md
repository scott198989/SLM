# HAVOC Tokenizer Guide

**Last Updated:** November 24, 2025
**Component:** Domain-Specialized Tokenizer for HAVOC-7B/SIGMA-7B

---

## Overview

The HAVOC tokenizer is a specialized SentencePiece-based tokenizer designed for mathematics, statistics, engineering, and manufacturing intelligence (DOE/SPC) domains. It provides comprehensive support for:

- **Mathematical notation** (Greek letters, operators, symbols)
- **Engineering units** (psi, MPa, °C, kW, etc.)
- **SRS reasoning markers** (8-stage reasoning pipeline)
- **DSL boundaries** (domain-specific language markers)
- **Tool invocations** (math and stats tool markers)
- **Normalized text processing** (Unicode → ASCII-safe where appropriate)

**Vocabulary size:** 70,000-80,000 tokens (configurable)

---

## Quick Start

### 1. Training a Tokenizer

```bash
# Train with default configuration (domain samples only)
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml

# Train with custom corpus
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml \
  --input-files data/math data/stats data/engineering

# Train with custom vocab size
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml \
  --vocab-size 80000
```

**Output:** Creates `artifacts/tokenizer/` containing:
- `tokenizer.model` - SentencePiece model
- `tokenizer.vocab` - Vocabulary file
- `tokenizer_metadata.json` - Special token metadata

### 2. Loading and Using

```python
from havoc_core.tokenizer import HavocTokenizer

# Load tokenizer
tokenizer = HavocTokenizer("artifacts/tokenizer")

# Encode text
text = "Calculate mean μ and variance σ²"
token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

# Decode tokens
decoded = tokenizer.decode(token_ids)

# Batch encoding
texts = ["First text", "Second text", "Third text"]
batch_ids = tokenizer.encode_batch(texts)

# Or use callable interface
token_ids = tokenizer(text)
```

### 3. Integration with Training Pipeline

```python
from havoc_core.tokenizer import HavocTokenizer
from havoc_data.dataset import CausalLMDataset

# Load tokenizer
tokenizer = HavocTokenizer("artifacts/tokenizer")

# Use in dataset
dataset = CausalLMDataset(
    tokenizer=tokenizer,
    sources=sources,
    mixture=mixture_config,
)
```

---

## Special Tokens

### Core Control Tokens (4)

- `<pad>` (ID=0) - Padding token
- `<bos>` (ID=1) - Beginning of sequence
- `<eos>` (ID=2) - End of sequence
- `<unk>` (ID=3) - Unknown token

### SRS Reasoning Stage Markers (8)

The Scott Reasoning Stack uses 8 stage markers:

```
<SRS_MODE>      → Problem identification
<SRS_GROUND>    → Knowledge grounding
<SRS_PLAN>      → Solution planning
<SRS_EXECUTE>   → Plan execution
<SRS_ARGUE>     → Generate counterarguments
<SRS_ARBITER>   → Resolve conflicts
<SRS_AUDIT>     → Verify solution
<SRS_ANSWER>    → Final answer
```

**Example usage:**
```
<SRS_MODE> Identify problem type: hypothesis testing <SRS_GROUND>
<SRS_GROUND> H0: μ₁ = μ₂, α = 0.05 <SRS_PLAN>
<SRS_PLAN> Perform t-test → analyze results → conclude <SRS_EXECUTE>
```

### DSL Markers (2)

For domain-specific language boundaries:

```
<DSL_BEGIN>     → Start of DSL expression
<DSL_END>       → End of DSL expression
```

**Example usage:**
```
<DSL_BEGIN> CHECK_SPC X-bar chart, n=5, UCL=100, LCL=80 <DSL_END>
<DSL_BEGIN> EVAL_DOE factorial 2³, RESPONSE=yield, FACTOR=temp,pressure <DSL_END>
```

### Tool Markers (2)

For tool invocation boundaries:

```
<TOOL_MATH>     → Math tool invocation
<TOOL_STATS>    → Statistical tool invocation
```

**Example usage:**
```
<TOOL_MATH> ∫₀¹ x² dx = 1/3 </TOOL_MATH>
<TOOL_STATS> ANOVA F(2,27)=5.43, p-value=0.011 </TOOL_STATS>
```

### Engineering Symbol Markers (2)

For engineering notation boundaries:

```
<ENG_SYMBOL_START>  → Start of engineering symbols
<ENG_SYMBOL_END>    → End of engineering symbols
```

**Example usage:**
```
<ENG_SYMBOL_START> σ = 250 MPa, ε = 0.02 <ENG_SYMBOL_END>
```

### Mathematical Symbols (14)

Preserved in vocabulary:

- `∑` (Sum), `∏` (Product)
- `∫` (Integral), `∂` (Partial derivative)
- `∇` (Gradient), `√` (Square root)
- `≈` (Approximately equal), `≠` (Not equal)
- `≤`, `≥` (Inequalities)
- `±` (Plus-minus)
- `×` (Multiplication), `÷` (Division)
- `∞` (Infinity)

### Greek Letters (48)

All lowercase and uppercase Greek letters:

- Lowercase: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω
- Uppercase: Α, Β, Γ, Δ, Ε, Ζ, Η, Θ, Ι, Κ, Λ, Μ, Ν, Ξ, Ο, Π, Ρ, Σ, Τ, Υ, Φ, Χ, Ψ, Ω

### Engineering Units (19)

Common engineering units as single tokens:

- Pressure: psi, MPa, GPa, kPa
- Force: N/m, N/cm, N/mm
- Density: kg/m³
- Velocity: m/s, m/s²
- Power: kW, MW, kWh
- Temperature: °C, °F, K
- Frequency: Hz, kHz, MHz, GHz

### DOE/SPC Domain Tokens (28+)

Statistical and manufacturing terms:

- **Statistical:** ANOVA, p-value, Cpk, Cp, Ppk, UCL, LCL, CL
- **DOE methods:** Box-Behnken, Taguchi, factorial, fractional_factorial, central_composite, Plackett-Burman
- **Charts:** control_chart, X-bar, R-chart
- **DSL commands:** CHECK_SPC, EVAL_DOE, RUN_TTEST, DEFINE_OPERATOR, FACTOR, RESPONSE

**Total special tokens:** 140+ (base + domain-specific)

---

## Text Normalization

The tokenizer applies intelligent normalization to handle mathematical and engineering text:

### Character-Level Normalization

**Superscripts → Caret notation:**
```
x² → x^2
x³ → x^3
```

**Subscripts → Underscore notation:**
```
H₂O → H_2O
μ₁ → μ_1
```

**Unicode dashes → Hyphen:**
```
− → -  (minus sign)
– → -  (en dash)
— → -  (em dash)
```

**Fractions → Slash notation:**
```
½ → 1/2
¼ → 1/4
¾ → 3/4
```

**Quotation marks → ASCII:**
```
" " → " "
' ' → ' '
```

### Whitespace Handling

- Multiple spaces → Single space
- Multiple hyphens → Single hyphen
- Preserves single spaces in math expressions

### What's NOT Normalized

The following are **preserved** for domain fidelity:

- Greek letters: μ, σ, α, β, etc.
- Math symbols: ∑, ∫, ∂, ∇, etc.
- Engineering units: °C, °F, MPa, etc.
- Inequalities: ≤, ≥, ≈, ≠

---

## Configuration

### YAML Configuration File

`configs/tokenizer/default.yaml`:

```yaml
vocab_size: 75000
model_type: bpe
character_coverage: 0.9995
max_sentence_length: 2048
normalize_text: true

special_tokens:
  - "<pad>"
  - "<bos>"
  - "<eos>"
  - "<unk>"
  - "<mask>"

input_files:
  - data/math
  - data/stats
  - data/engineering
  - data/general

output_dir: artifacts/tokenizer
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 75000 | Vocabulary size (70k-80k recommended) |
| `model_type` | str | "bpe" | SentencePiece type: bpe, unigram, char, word |
| `character_coverage` | float | 0.9995 | Char coverage (high for math symbols) |
| `max_sentence_length` | int | 2048 | Max chars per sentence |
| `normalize_text` | bool | true | Apply math-aware normalization |
| `special_tokens` | list | [...] | Base special tokens |
| `input_files` | list | [] | Corpus files/directories |
| `output_dir` | str | artifacts/tokenizer | Output directory |

### Command-Line Overrides

```bash
# Override vocab size
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml \
  --vocab-size 80000

# Override output directory
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml \
  --output-dir custom_tokenizer

# Override input files
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml \
  --input-files corpus1.txt corpus2.txt data/math

# Override model type
python -m havoc_core.tokenizer.train_tokenizer \
  --config configs/tokenizer/default.yaml \
  --model-type unigram
```

---

## API Reference

### `HavocTokenizer` Class

#### Initialization

```python
from havoc_core.tokenizer import HavocTokenizer

# From directory
tokenizer = HavocTokenizer("artifacts/tokenizer")

# From model file
tokenizer = HavocTokenizer("artifacts/tokenizer/tokenizer.model")

# With explicit metadata
tokenizer = HavocTokenizer(
    model_path="path/to/tokenizer.model",
    metadata_path="path/to/metadata.json"
)
```

#### Properties

```python
tokenizer.vocab_size       # Vocabulary size
tokenizer.pad_id          # Padding token ID (0)
tokenizer.bos_id          # BOS token ID (1)
tokenizer.eos_id          # EOS token ID (2)
tokenizer.unk_id          # Unknown token ID (3)
tokenizer.special_tokens  # Dict[str, int] of special tokens
```

#### Methods

**Encoding:**

```python
# Basic encoding
token_ids = tokenizer.encode("text")

# With BOS/EOS
token_ids = tokenizer.encode("text", add_bos=True, add_eos=True)

# Batch encoding
batch_ids = tokenizer.encode_batch(["text1", "text2", "text3"])

# Callable interface
token_ids = tokenizer("text")
batch_ids = tokenizer(["text1", "text2"])
```

**Decoding:**

```python
# Decode single sequence
text = tokenizer.decode(token_ids)

# Decode batch
texts = tokenizer.decode([ids1, ids2, ids3])

# Skip special tokens
text = tokenizer.decode(token_ids, skip_special_tokens=True)
```

**Token Access:**

```python
# Get token ID
token_id = tokenizer.sp.PieceToId("ANOVA")

# Get token string
token_str = tokenizer.sp.IdToPiece(1234)

# Get special token ID
srs_mode_id = tokenizer.get_special_token_id("<SRS_MODE>")

# Get all SRS token IDs
srs_ids = tokenizer.get_srs_token_ids()

# Get DSL token IDs
dsl_ids = tokenizer.get_dsl_token_ids()

# Get tool token IDs
tool_ids = tokenizer.get_tool_token_ids()
```

**Tokenization (for debugging):**

```python
# Get token pieces
pieces = tokenizer.tokenize("Calculate μ and σ²")
# ['▁Calculate', '▁', 'μ', '▁and', '▁', 'σ', '^', '2']
```

### `train_tokenizer` Function

```python
from havoc_core.tokenizer import train_tokenizer
from havoc_core.config import TokenizerTrainingConfig

config = TokenizerTrainingConfig(
    vocab_size=75000,
    model_type="bpe",
    input_files=["data/math", "data/stats"],
    output_dir="artifacts/tokenizer",
)

metadata = train_tokenizer(config, verbose=True)

print(metadata.vocab_size)       # 75000
print(metadata.special_tokens)   # List of all special tokens
print(metadata.domain_tokens)    # List of domain-specific tokens
```

---

## Usage Examples

### Example 1: Math Expression

```python
tokenizer = HavocTokenizer("artifacts/tokenizer")

text = "Calculate mean μ = ∑ xᵢ / n and variance σ² = ∑(xᵢ - μ)² / n"

# Encode
token_ids = tokenizer.encode(text)
print(f"Tokens: {len(token_ids)}")

# Decode
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
```

### Example 2: SRS Reasoning Pipeline

```python
text = """
<SRS_MODE> Identify problem type: hypothesis testing <SRS_GROUND>
<SRS_GROUND> Given: Two samples A and B, test if means differ <SRS_PLAN>
<SRS_PLAN> Use two-sample t-test with α = 0.05 <SRS_EXECUTE>
<SRS_EXECUTE> t = 2.456, p-value = 0.023 <SRS_ARGUE>
<SRS_ARGUE> Could be Type I error, check assumptions <SRS_ARBITER>
<SRS_ARBITER> Normality OK, equal variance OK, reject H0 <SRS_AUDIT>
<SRS_AUDIT> Verified: p < α, conclusion valid <SRS_ANSWER>
<SRS_ANSWER> Reject H0: means are significantly different
"""

token_ids = tokenizer.encode(text.strip())
decoded = tokenizer.decode(token_ids)
```

### Example 3: DSL Expression

```python
text = """
<DSL_BEGIN>
CHECK_SPC X-bar chart
  n = 5
  UCL = 100.5
  LCL = 99.5
  CL = 100.0
<DSL_END>
"""

token_ids = tokenizer.encode(text.strip())

# Check for DSL markers
dsl_ids = tokenizer.get_dsl_token_ids()
print(f"DSL_BEGIN ID: {dsl_ids['<DSL_BEGIN>']}")
print(f"DSL_END ID: {dsl_ids['<DSL_END>']}")
```

### Example 4: Engineering Symbols

```python
text = """
<ENG_SYMBOL_START>
Material: Steel AISI 1045
Yield strength: σ_y = 530 MPa
Ultimate tensile strength: σ_u = 625 MPa
Young's modulus: E = 200 GPa
Operating temperature: T = 150°C
Applied stress: σ = 250 MPa
Safety factor: n = σ_y / σ = 2.12
<ENG_SYMBOL_END>
"""

token_ids = tokenizer.encode(text.strip())
decoded = tokenizer.decode(token_ids)
```

### Example 5: Mixed Content

```python
text = """
Perform ANOVA on Box-Behnken design:
- Factors: temperature (100-200°C), pressure (5-15 MPa), time (1-5 h)
- Response: yield (%)
- α = 0.05

<TOOL_STATS>
ANOVA results:
  F(2, 15) = 8.42
  p-value = 0.003
  Conclusion: At least one factor is significant
</TOOL_STATS>

<SRS_ANSWER>
Temperature and pressure are significant (p < 0.05).
Optimal conditions: T = 175°C, P = 10 MPa, t = 3 h
Expected yield: 92.5% ± 1.2% (95% CI)
</SRS_ANSWER>
"""

token_ids = tokenizer.encode(text.strip())
print(f"Total tokens: {len(token_ids)}")
```

---

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
# All tests
pytest tests/test_tokenizer.py -v

# Specific test class
pytest tests/test_tokenizer.py::TestVocabUtils -v
pytest tests/test_tokenizer.py::TestMathExpressionTokenization -v

# With coverage
pytest tests/test_tokenizer.py --cov=havoc_core.tokenizer
```

### Quick Test Script

```bash
# Run quick validation test
python test_tokenizer_quick.py
```

This tests:
- Special token registration
- Tokenizer training
- Loading and inference
- Math symbol handling
- Encoding/decoding

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sentencepiece'"

**Solution:** Install SentencePiece:
```bash
pip install sentencepiece>=0.1.99
```

Or install the full package:
```bash
pip install -e .
```

### Issue: Tokenizer training is slow

**Causes:**
- Large corpus
- High vocab size

**Solutions:**
- Start with smaller corpus for testing
- Use `vocab_size=5000-10000` for quick tests
- For production, use full corpus with `vocab_size=75000`

### Issue: Math symbols not tokenizing correctly

**Check:**
1. Character coverage is high enough (`character_coverage: 0.9995`)
2. Normalization is enabled (`normalize_text: true`)
3. Domain samples are included in training corpus

### Issue: SRS/DSL tokens not recognized

**Solution:** Ensure tokenizer was trained with domain samples:
```python
from havoc_core.tokenizer.vocab_utils import sample_domain_strings

# Domain samples include SRS/DSL examples
samples = sample_domain_strings()
print(samples)
```

These are automatically included during training.

### Issue: Tokenizer output different from input

**Expected behavior:** Some normalization is intentional:
- Superscripts → caret notation (x² → x^2)
- Subscripts → underscore notation (H₂O → H_2O)
- Multiple spaces → single space

**To disable normalization:**
```yaml
# In config YAML
normalize_text: false
```

---

## Best Practices

### 1. Corpus Preparation

✅ **Do:**
- Include diverse examples of target domain
- Add math/stats expressions with Greek letters and symbols
- Include engineering calculations with units
- Add SRS reasoning examples
- Include DSL command samples

❌ **Don't:**
- Use only general text without domain content
- Skip math symbol examples
- Forget to include special token usage examples

### 2. Vocabulary Size

| Model Size | Recommended Vocab Size |
|------------|------------------------|
| 1-3B | 50,000 - 65,000 |
| 7B | 70,000 - 80,000 |
| 13B+ | 80,000 - 100,000 |

**Trade-offs:**
- Larger vocab: Better domain coverage, larger model
- Smaller vocab: Faster, but may struggle with rare terms

### 3. Character Coverage

For specialized domains with math symbols:
```yaml
character_coverage: 0.9995  # Recommended
```

For general text only:
```yaml
character_coverage: 0.999   # Sufficient
```

### 4. Model Type

- **BPE (recommended):** Balanced, good for most use cases
- **Unigram:** More flexible, good for morphologically rich languages
- **Word:** Fast, but vocabulary explosion risk
- **Char:** Compact, but long sequences

### 5. Integration with Training

After training tokenizer:

1. Update training config:
```yaml
# configs/training/default_training.yaml
tokenizer_path: artifacts/tokenizer
```

2. Load in training script:
```python
from havoc_core.tokenizer import HavocTokenizer

tokenizer = HavocTokenizer(config.tokenizer_path)
dataset = CausalLMDataset(tokenizer, sources, mixture)
```

---

## Performance

### Training Time

| Corpus Size | Vocab Size | Approx. Time |
|-------------|------------|--------------|
| 1 MB | 10,000 | ~10 seconds |
| 10 MB | 50,000 | ~1 minute |
| 100 MB | 75,000 | ~5 minutes |
| 1 GB | 75,000 | ~30 minutes |

### Inference Speed

- Encoding: ~100K tokens/second (CPU)
- Decoding: ~50K tokens/second (CPU)
- Batch encoding: ~500K tokens/second (batches of 32)

---

## Future Enhancements

Planned features:

- [ ] Byte-level BPE support
- [ ] Custom pre-tokenization rules
- [ ] Multi-lingual support for engineering domains
- [ ] Automatic corpus augmentation
- [ ] Tokenizer compression for deployment
- [ ] Integration with Hugging Face tokenizers library

---

## References

- [SentencePiece Documentation](https://github.com/google/sentencepiece)
- [BPE Algorithm](https://arxiv.org/abs/1508.07909)
- [Unicode Normalization](https://unicode.org/reports/tr15/)

---

**For questions or issues:** Check existing tests in `tests/test_tokenizer.py` or refer to sample code in `test_tokenizer_quick.py`.

**Last updated:** November 24, 2025
