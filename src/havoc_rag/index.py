from __future__ import annotations

from typing import List, Tuple

import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[dict] = []

    def add(self, embeddings: np.ndarray, metadatas: List[dict]) -> None:
        for vec, meta in zip(embeddings, metadatas):
            self.vectors.append(vec)
            self.metadata.append(meta)

    def query(self, query: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        scores = []
        for vec, meta in zip(self.vectors, self.metadata):
            score = -np.linalg.norm(query - vec)
            scores.append((meta, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
