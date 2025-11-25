from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass
class DataSource:
    """
    Abstract data source. Subclasses must implement stream_documents().
    """

    name: str
    paths: List[str] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self) -> None:
        # Support single-string path for backward compatibility
        if isinstance(self.paths, str):
            self.paths = [self.paths]

    def files(self, glob: str = "*.txt") -> List[Path]:
        collected: List[Path] = []
        for path in self.paths:
            p = Path(path)
            if p.is_dir():
                collected.extend(p.rglob(glob))
            elif p.is_file():
                collected.append(p)
        return collected

    def stream_documents(self) -> Iterator[str]:
        raise NotImplementedError


class TextFileSource(DataSource):
    """Plain text files (one example per line)."""

    def stream_documents(self) -> Iterator[str]:
        for file_path in self.files("*.txt"):
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip("\n")
                    if stripped:
                        yield stripped


class JSONLSource(DataSource):
    """JSONL source that extracts a chosen text field per object."""

    def __init__(self, name: str, paths: List[str], weight: float = 1.0, text_field: str = "text"):
        super().__init__(name=name, paths=paths, weight=weight)
        self.text_field = text_field

    def stream_documents(self) -> Iterator[str]:
        for file_path in self.files("*.jsonl"):
            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        obj = json.loads(line)
                        text = obj.get(self.text_field) or obj.get("content") or obj.get("body")
                        if text:
                            yield str(text)
                    except Exception:
                        continue


class EngineeringCorpusSource(TextFileSource):
    """Engineering-heavy corpus. Reuses text files but allows different weighting."""


class StatisticsCorpusSource(TextFileSource):
    """Statistics/math-heavy corpus."""


class ManufacturingCorpusSource(TextFileSource):
    """Manufacturing/operations corpus."""


class InMemorySource(DataSource):
    """
    Utility source for tests and synthetic data. Cycles through provided items.
    """

    def __init__(self, name: str, items: Iterable[str], weight: float = 1.0):
        super().__init__(name=name, paths=[], weight=weight)
        self._items = list(items)
        self._cycle = cycle(self._items) if self._items else None

    def stream_documents(self) -> Iterator[str]:
        if not self._cycle:
            return iter([])
        while True:
            yield next(self._cycle)


def build_source_from_dict(defn: dict) -> DataSource:
    """
    Instantiate a DataSource from a YAML/JSON dictionary definition.
    """
    name = defn.get("name", "source")
    source_type = (defn.get("type") or "text").lower()
    weight = float(defn.get("weight", 1.0))
    paths = defn.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]

    text_field = defn.get("text_field", "text")

    if source_type == "jsonl":
        return JSONLSource(name=name, paths=paths, weight=weight, text_field=text_field)
    if source_type == "engineering":
        return EngineeringCorpusSource(name=name, paths=paths, weight=weight)
    if source_type == "statistics":
        return StatisticsCorpusSource(name=name, paths=paths, weight=weight)
    if source_type == "manufacturing":
        return ManufacturingCorpusSource(name=name, paths=paths, weight=weight)
    return TextFileSource(name=name, paths=paths, weight=weight)


def load_sources(definitions: List[dict]) -> List[DataSource]:
    return [build_source_from_dict(d) for d in definitions]
