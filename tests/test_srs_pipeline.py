import pytest

pytest.importorskip("statsmodels")

from havoc_core.config import SRSConfig
from havoc_srs.orchestrator import run_pipeline


def test_pipeline_runs():
    answer = run_pipeline("Run a t-test on two samples", SRSConfig())
    assert answer.conclusion
    assert answer.confidence >= 0.0
