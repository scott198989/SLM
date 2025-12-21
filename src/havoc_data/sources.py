from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class DataSource:
    """Represents a source of training data.

    Attributes:
        name: Name of this data source (e.g., "math", "engineering")
        paths: List of file or directory paths
        weight: Sampling weight for mixture (higher = more likely to sample)
        domain: Optional domain tag (e.g., "mathematics", "statistics", "engineering")
        file_type: Type of files to load ("txt", "jsonl", or "auto")
        text_field: For JSONL, which field contains the text (default: "text")
    """

    name: str
    paths: List[str]
    weight: float = 1.0
    domain: Optional[str] = None
    file_type: str = "auto"  # "txt", "jsonl", or "auto"
    text_field: str = "text"  # For JSONL files
    metadata_fields: List[str] = field(default_factory=list)  # Additional fields to extract

    def files(self) -> List[Path]:
        """Collect all matching files from configured paths.

        Returns:
            List of Path objects for files to process
        """
        collected = []
        for path_str in self.paths:
            p = Path(path_str)

            if not p.exists():
                continue

            if p.is_dir():
                # Collect files based on type
                if self.file_type == "txt" or self.file_type == "auto":
                    collected.extend(p.rglob("*.txt"))
                if self.file_type == "jsonl" or self.file_type == "auto":
                    collected.extend(p.rglob("*.jsonl"))
                    collected.extend(p.rglob("*.json"))
            elif p.is_file():
                collected.append(p)

        return collected

    def iter_documents(self, max_docs: Optional[int] = None) -> Iterator[dict]:
        """Iterate over documents from this source.

        Yields:
            Dict with keys:
                - text: The main text content
                - source: Name of this data source
                - domain: Domain tag if specified
                - file_path: Path to source file
                - metadata: Additional metadata from JSONL if present
        """
        doc_count = 0

        for file_path in self.files():
            # Determine file type
            file_type = self._detect_file_type(file_path)

            if file_type == "jsonl":
                yield from self._iter_jsonl(file_path, max_docs, doc_count)
            else:
                yield from self._iter_text(file_path, max_docs, doc_count)

            if max_docs and doc_count >= max_docs:
                break

    def _detect_file_type(self, path: Path) -> str:
        """Detect file type from extension or config."""
        if self.file_type != "auto":
            return self.file_type

        suffix = path.suffix.lower()
        if suffix in [".jsonl", ".json"]:
            return "jsonl"
        return "txt"

    def _iter_jsonl(
        self, file_path: Path, max_docs: Optional[int], doc_count: int
    ) -> Iterator[dict]:
        """Iterate over JSONL file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_docs and doc_count >= max_docs:
                        break

                    try:
                        data = json.loads(line.strip())

                        # Extract text field
                        text = data.get(self.text_field, "")
                        if not text:
                            continue

                        # Extract metadata
                        metadata = {}
                        for field in self.metadata_fields:
                            if field in data:
                                metadata[field] = data[field]

                        yield {
                            "text": text,
                            "source": self.name,
                            "domain": self.domain,
                            "file_path": str(file_path),
                            "metadata": metadata,
                        }

                        doc_count += 1

                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

        except Exception as e:
            # Log error but don't crash
            print(f"Warning: Failed to read {file_path}: {e}")

    def _iter_text(
        self, file_path: Path, max_docs: Optional[int], doc_count: int
    ) -> Iterator[dict]:
        """Iterate over plain text file.

        Yields one document per file (full file content).
        """
        if max_docs and doc_count >= max_docs:
            return

        try:
            text = file_path.read_text(encoding="utf-8")

            yield {
                "text": text,
                "source": self.name,
                "domain": self.domain,
                "file_path": str(file_path),
                "metadata": {},
            }

        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")

    def __repr__(self) -> str:
        return f"DataSource(name={self.name}, domain={self.domain}, weight={self.weight}, files={len(self.files())})"
