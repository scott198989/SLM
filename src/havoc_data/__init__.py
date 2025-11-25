"""HAVOC-7B Data Module

This module handles dataset loading, preprocessing, and data source management.
"""

from havoc_data.build import build_dataset
from havoc_data.dataset import CausalLMDataset
from havoc_data.preprocess import annotate_reasoning, iter_normalized, normalize_text, normalize_symbols, tag_dsl
from havoc_data.sources import (
    DataSource,
    EngineeringCorpusSource,
    InMemorySource,
    JSONLSource,
    ManufacturingCorpusSource,
    StatisticsCorpusSource,
    TextFileSource,
    build_source_from_dict,
    load_sources,
)

__all__ = [
    "CausalLMDataset",
    "build_dataset",
    "DataSource",
    "TextFileSource",
    "JSONLSource",
    "EngineeringCorpusSource",
    "StatisticsCorpusSource",
    "ManufacturingCorpusSource",
    "InMemorySource",
    "build_source_from_dict",
    "load_sources",
    "annotate_reasoning",
    "iter_normalized",
    "normalize_text",
    "normalize_symbols",
    "tag_dsl",
]
