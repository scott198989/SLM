from __future__ import annotations

from typing import List

from havoc_rag.embeddings import EmbeddingModel
from havoc_rag.index import VectorIndex


class Retriever:
    def __init__(self, embed_dim: int = 768):
        self.embedder = EmbeddingModel(dim=embed_dim)
        self.index = VectorIndex(dim=embed_dim)

    def add_corpus(self, texts: List[str]) -> None:
        embeddings = self.embedder.embed_texts(texts)
        metadatas = [{"text": text} for text in texts]
        self.index.add(embeddings, metadatas)

    def retrieve_references(self, query: str, k: int = 5) -> List[dict]:
        query_embedding = self.embedder.embed_texts([query])[0]
        results = self.index.query(query_embedding, k=k)
        return [{"metadata": meta, "score": score} for meta, score in results]
