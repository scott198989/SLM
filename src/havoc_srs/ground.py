from __future__ import annotations

from dataclasses import dataclass
from typing import List

from havoc_rag.retrieval import Retriever
from havoc_srs.mode import ModePrediction


@dataclass
class GroundedContext:
    references: List[dict]
    retriever: Retriever


def attach_references(prompt: str, mode: ModePrediction, retriever: Retriever) -> GroundedContext:
    refs = retriever.retrieve_references(prompt, k=5)
    return GroundedContext(references=refs, retriever=retriever)
