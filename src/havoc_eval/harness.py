from __future__ import annotations

from dataclasses import dataclass
from typing import List

from havoc_core.config import EvalConfig, SRSConfig
from havoc_srs.orchestrator import run_pipeline
from havoc_eval.benchmarks import BenchmarkItem, default_benchmarks


@dataclass
class EvaluationResult:
    item: BenchmarkItem
    output: str
    confidence: float


def run_evaluation(config: EvalConfig, srs_config: SRSConfig) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for item in default_benchmarks():
        answer = run_pipeline(item.prompt, srs_config)
        results.append(EvaluationResult(item=item, output=answer.conclusion, confidence=answer.confidence))
    return results
