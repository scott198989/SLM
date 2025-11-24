from __future__ import annotations

import numpy as np


class EmbeddingModel:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_texts(self, texts):
        # Placeholder: deterministic random embeddings based on hash
        vecs = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vecs.append(rng.standard_normal(self.dim))
        return np.vstack(vecs)
