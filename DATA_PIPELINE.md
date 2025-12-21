# HAVOC-7B Data Pipeline Guide

This guide explains how to use the complete data pipeline for training HAVOC-7B.

## Quick Start

```bash
# 1. Build and validate dataset
python -m havoc_data.build --config configs/data/default_data.yaml --validate --show-mixture

# 2. Test with sample data
python -m havoc_data.build --config configs/data/default_data.yaml --test-samples 10

# 3. Use in training
python scripts/train.py --data-config configs/data/default_data.yaml
```

## Features Implemented

✅ **Streaming Dataset**: `CausalLMDataset` is an `IterableDataset` for memory-efficient training
✅ **Mixture Weighting**: Sample from multiple sources according to configurable weights
✅ **Sample Packing**: Optionally pack multiple short documents into single sequences
✅ **BOS/EOS Handling**: Proper beginning and end-of-sequence token management
✅ **Preprocessing**: Symbol normalization, DSL extraction, reasoning trace annotation
✅ **Multiple Formats**: Supports `.txt` and `.jsonl` files
✅ **Domain Tagging**: Tag data sources by domain (math, stats, engineering, etc.)

## Data Directory Structure

```
data/
├── math/                    # Mathematics content (weight: 3.0)
│   ├── example_calculus.txt
│   └── example_linear_algebra.txt
├── stats/                   # Statistics content (weight: 3.0)
│   ├── example_hypothesis_testing.txt
│   └── example_regression.txt
├── engineering/             # Engineering content (weight: 3.0)
│   └── example_thermodynamics.txt
├── manufacturing/           # DOE/SPC content (weight: 2.0)
│   ├── example_doe.txt
│   └── example_spc.txt
├── general/                 # General knowledge (weight: 2.0)
│   └── example_algorithms.txt
├── dialog/                  # Conversational data (weight: 1.0)
│   └── example_conversation.txt
├── dsl/                     # DSL examples (weight: 0.5)
│   └── example_dsl.txt
└── reasoning/               # Reasoning traces (weight: 0.5)
    └── example_reasoning.txt
```

## Configuration

Edit `configs/data/default_data.yaml`:

```yaml
mixture:
  domain_ratio: 0.6        # Ratio of domain-specific content
  general_ratio: 0.3       # Ratio of general content
  dialog_ratio: 0.1        # Ratio of dialog content
  max_sequence_length: 4096

sources:
  - name: mathematics
    paths: [data/math/]
    weight: 3.0            # Higher weight = sampled more often
    domain: mathematics
    file_type: auto        # "txt", "jsonl", or "auto"
```

## Using with Trainer

```python
from havoc_data import CausalLMDataset, causal_lm_collate_fn
from havoc_data.build import build_sources_from_config, build_mixture_from_config
from torch.utils.data import DataLoader

# Load config
config = load_config("configs/data/default_data.yaml")
sources = build_sources_from_config(config)
mixture = build_mixture_from_config(config)

# Create dataset
dataset = CausalLMDataset(
    tokenizer=tokenizer,
    sources=sources,
    mixture=mixture,
    max_seq_len=4096,
    enable_packing=False,  # Set to True for sample packing
)

# Create DataLoader for Trainer
# IMPORTANT: Use shuffle=False for IterableDataset
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,  # IterableDataset doesn't support shuffling
    collate_fn=causal_lm_collate_fn,
    num_workers=0,
)

# Use with Trainer
trainer = Trainer(
    config=training_config,
    model=model,
    train_dataset=dataset,
)
```

## Preprocessing Features

### Symbol Normalization

Unicode math/engineering symbols are converted to ASCII:
- `σ` → `sigma`
- `μ` → `mu`
- `°` → `deg`
- `±` → `+/-`
- And 20+ more symbols

### DSL Extraction

DSL code blocks are tagged:

```
Input:
```dsl
EXPERIMENT test
  FACTORS a, b
END
```

Output:
<DSL_BEGIN>
EXPERIMENT test
  FACTORS a, b
END
<DSL_END>
```

### Reasoning Trace Annotation

Reasoning traces are structured:

```
Input:
<think>Calculate the mean...</think>

Output:
<REASONING_BEGIN>
Calculate the mean...
<REASONING_END>
```

## JSONL Support

For JSONL files:

```yaml
sources:
  - name: arxiv_papers
    paths: [data/arxiv/papers.jsonl]
    weight: 1.5
    file_type: jsonl
    text_field: abstract      # Which field contains text
    metadata_fields: [title, authors]  # Additional fields to extract
```

JSONL format:
```json
{"text": "Paper abstract...", "title": "Paper Title", "authors": ["Author 1"]}
```

## Mixture Statistics

View sampling probabilities:

```bash
python -m havoc_data.build --config configs/data/default_data.yaml --show-mixture
```

Output:
```
MIXTURE STATISTICS
==================
mathematics: 20.00%
statistics: 20.00%
engineering: 20.00%
manufacturing: 13.33%
general: 13.33%
dialog: 6.67%
dsl_examples: 3.33%
reasoning_traces: 3.33%
```

## Validation

Check that all data sources are valid:

```bash
python -m havoc_data.build --config configs/data/default_data.yaml --validate
```

Output shows:
- Number of sources
- Number of files per source
- Which sources are valid/invalid
- Total file count

## Testing

Run comprehensive tests:

```bash
pytest tests/test_data_pipeline.py -v
```

Tests cover:
- Symbol normalization
- DSL extraction
- Reasoning annotation
- Malformed line detection
- DataSource file discovery
- JSONL parsing
- Mixture weighting
- Sample packing
- BOS/EOS handling
- Padding and masking

## Performance Tips

1. **Enable Sample Packing**: Reduces padding for short documents
   ```python
   dataset = CausalLMDataset(..., enable_packing=True)
   ```

2. **Adjust Mixture Weights**: Balance your data distribution
   - High weight for critical domains
   - Low weight for supplementary data

3. **Use JSONL for Large Datasets**: More efficient than plain text for structured data

4. **Monitor Mixture Stats**: Ensure sampling matches your expectations

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- Solution: Install PyTorch: `pip install torch`

**Issue**: No files found in data sources
- Solution: Check paths in config match your data directory structure

**Issue**: "IterableDataset doesn't support len()"
- Solution: This is normal. Use `max_steps` in training config instead of epochs

**Issue**: Out of memory during training
- Solution: Reduce `batch_size` or enable `sample_packing` to reduce padding

## Example Workflow

```bash
# 1. Prepare your data
mkdir -p data/math data/stats data/engineering
# Add .txt files to each directory

# 2. Validate configuration
python -m havoc_data.build --config configs/data/default_data.yaml --validate

# 3. Test dataset
python -m havoc_data.build --config configs/data/default_data.yaml --test-samples 5

# 4. Train model
python scripts/train.py \
    --config configs/training/default_training.yaml \
    --data-config configs/data/default_data.yaml \
    --max-steps 10000
```

## API Reference

### `CausalLMDataset`

Main dataset class for causal language modeling.

**Parameters**:
- `tokenizer`: Tokenizer with `encode()` method
- `sources`: List of `DataSource` objects
- `mixture`: `DataMixtureConfig` instance
- `max_seq_len`: Maximum sequence length (default: 4096)
- `bos_token_id`: BOS token ID (default: 1)
- `eos_token_id`: EOS token ID (default: 2)
- `pad_token_id`: Padding token ID (default: 0)
- `enable_packing`: Enable sample packing (default: False)
- `extract_dsl`: Extract DSL blocks (default: True)
- `annotate_reasoning`: Annotate reasoning traces (default: True)

### `DataSource`

Represents a data source with files and sampling weight.

**Parameters**:
- `name`: Source name (e.g., "mathematics")
- `paths`: List of file/directory paths
- `weight`: Sampling weight (default: 1.0)
- `domain`: Optional domain tag
- `file_type`: "txt", "jsonl", or "auto" (default: "auto")
- `text_field`: JSONL text field (default: "text")
- `metadata_fields`: Additional JSONL fields to extract

### `MixturePolicy`

Handles weighted sampling from multiple sources.

**Methods**:
- `sample()`: Sample a data source according to weights
- `get_mixture_stats()`: Get sampling probability statistics

### Collate Functions

- `causal_lm_collate_fn(batch)`: Returns (input_ids, attention_mask)
- `causal_lm_collate_fn_with_labels(batch)`: Returns (input_ids, attention_mask, labels)

## Next Steps

1. Add your own data to the `data/` directories
2. Adjust mixture weights in config
3. Train tokenizer: `python -m havoc_core.tokenizer.train_tokenizer`
4. Start training: `python scripts/train.py`

For more information, see:
- `TRAINING_GUIDE.md` - Training workflow
- `README.md` - Project overview
- `CLAUDE.md` - Development guide
