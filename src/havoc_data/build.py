"""Build and validate datasets for HAVOC-7B training.

This module provides utilities for building datasets from configuration files,
validating data sources, and testing the data pipeline.

Usage:
    python -m havoc_data.build --config configs/data/default_data.yaml
    python -m havoc_data.build --config configs/data/default_data.yaml --validate
    python -m havoc_data.build --config configs/data/default_data.yaml --test-samples 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml

from havoc_core.config import DataMixtureConfig
from havoc_data.dataset import CausalLMDataset
from havoc_data.sources import DataSource


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict with config data
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_sources_from_config(config: dict) -> list[DataSource]:
    """Build DataSource objects from config.

    Args:
        config: Config dict with 'sources' key

    Returns:
        List of DataSource instances
    """
    sources = []

    for source_config in config.get("sources", []):
        source = DataSource(
            name=source_config["name"],
            paths=source_config["paths"],
            weight=source_config.get("weight", 1.0),
            domain=source_config.get("domain"),
            file_type=source_config.get("file_type", "auto"),
            text_field=source_config.get("text_field", "text"),
            metadata_fields=source_config.get("metadata_fields", []),
        )
        sources.append(source)

    return sources


def build_mixture_from_config(config: dict) -> DataMixtureConfig:
    """Build DataMixtureConfig from config.

    Args:
        config: Config dict with 'mixture' key

    Returns:
        DataMixtureConfig instance
    """
    mixture_config = config.get("mixture", {})

    return DataMixtureConfig(
        domain_ratio=mixture_config.get("domain_ratio", 0.6),
        general_ratio=mixture_config.get("general_ratio", 0.3),
        dialog_ratio=mixture_config.get("dialog_ratio", 0.1),
        max_sequence_length=mixture_config.get("max_sequence_length", 4096),
    )


def validate_sources(sources: list[DataSource]) -> dict:
    """Validate data sources and report statistics.

    Args:
        sources: List of DataSource instances

    Returns:
        Dict with validation results
    """
    results = {
        "total_sources": len(sources),
        "valid_sources": 0,
        "total_files": 0,
        "sources_detail": [],
    }

    for source in sources:
        files = source.files()
        file_count = len(files)

        source_info = {
            "name": source.name,
            "domain": source.domain,
            "weight": source.weight,
            "file_count": file_count,
            "paths": source.paths,
            "valid": file_count > 0,
        }

        if file_count > 0:
            results["valid_sources"] += 1
            results["total_files"] += file_count

        results["sources_detail"].append(source_info)

    return results


def print_validation_report(results: dict):
    """Print validation report to console."""
    print("\n" + "=" * 70)
    print("DATA SOURCE VALIDATION REPORT")
    print("=" * 70)
    print(f"\nTotal sources: {results['total_sources']}")
    print(f"Valid sources: {results['valid_sources']}")
    print(f"Total files: {results['total_files']}")

    print("\n" + "-" * 70)
    print("Source Details:")
    print("-" * 70)

    for source_info in results["sources_detail"]:
        status = "✓ VALID" if source_info["valid"] else "✗ INVALID"
        print(f"\n{status} - {source_info['name']}")
        print(f"  Domain: {source_info['domain']}")
        print(f"  Weight: {source_info['weight']}")
        print(f"  Files: {source_info['file_count']}")
        print(f"  Paths: {', '.join(source_info['paths'])}")

    print("\n" + "=" * 70)


def test_dataset(sources: list[DataSource], mixture: DataMixtureConfig, num_samples: int = 10):
    """Test the dataset by sampling a few examples.

    Args:
        sources: List of DataSource instances
        mixture: DataMixtureConfig
        num_samples: Number of samples to test
    """
    print("\n" + "=" * 70)
    print("DATASET TEST")
    print("=" * 70)

    # Create a dummy tokenizer for testing
    class DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            # Simple word-based tokenization for testing
            return [hash(word) % 10000 for word in text.split()]

        def decode(self, tokens: list[int]) -> str:
            return " ".join(str(t) for t in tokens)

    tokenizer = DummyTokenizer()

    # Create dataset
    print("\nCreating dataset...")
    dataset = CausalLMDataset(
        tokenizer=tokenizer,
        sources=sources,
        mixture=mixture,
        max_seq_len=mixture.max_sequence_length,
        enable_packing=False,
    )

    # Sample examples
    print(f"\nSampling {num_samples} examples...")
    iterator = iter(dataset)

    for i in range(num_samples):
        try:
            example = next(iterator)
            print(f"\n--- Example {i + 1} ---")
            print(f"Input IDs shape: {example.input_ids.shape}")
            print(f"Attention mask shape: {example.attention_mask.shape}")
            print(f"Labels shape: {example.labels.shape}")
            print(f"Non-padding tokens: {example.attention_mask.sum().item()}")
            print(f"First 10 tokens: {example.input_ids[:10].tolist()}")

        except StopIteration:
            print(f"\nDataset exhausted after {i} examples")
            break
        except Exception as e:
            print(f"\nError sampling example {i + 1}: {e}")
            break

    print("\n" + "=" * 70)


def main():
    """Main entry point for dataset building."""
    parser = argparse.ArgumentParser(description="Build HAVOC-7B dataset from config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/default_data.yaml",
        help="Path to data configuration YAML file",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate data sources and print report"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="Number of samples to test (0 = no testing)",
    )
    parser.add_argument(
        "--show-mixture", action="store_true", help="Show mixture statistics"
    )

    args = parser.parse_args()

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    print(f"Loading config from: {args.config}")

    # Load configuration
    config = load_config(args.config)

    # Build sources
    print("\nBuilding data sources...")
    sources = build_sources_from_config(config)
    print(f"Loaded {len(sources)} data sources")

    # Build mixture config
    mixture = build_mixture_from_config(config)

    # Validate if requested
    if args.validate or args.test_samples == 0:
        results = validate_sources(sources)
        print_validation_report(results)

        if results["valid_sources"] == 0:
            print("\n⚠ Warning: No valid sources found!")
            return 1

    # Show mixture stats if requested
    if args.show_mixture:
        from havoc_data.dataset import MixturePolicy

        policy = MixturePolicy(sources)
        stats = policy.get_mixture_stats()

        print("\n" + "=" * 70)
        print("MIXTURE STATISTICS")
        print("=" * 70)
        for source_name, prob in stats.items():
            print(f"{source_name}: {prob:.2%}")
        print("=" * 70)

    # Test dataset if requested
    if args.test_samples > 0:
        test_dataset(sources, mixture, args.test_samples)

    print("\n✓ Dataset build complete!")
    return 0


if __name__ == "__main__":
    exit(main())
