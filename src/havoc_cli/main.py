from __future__ import annotations

import argparse

from havoc_core.config import SRSConfig
from havoc_srs.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HAVOC-7B reasoning pipeline")
    parser.add_argument("prompt", type=str, help="User prompt")
    args = parser.parse_args()

    srs_config = SRSConfig()
    answer = run_pipeline(args.prompt, srs_config)
    print("Conclusion:", answer.conclusion)
    print("Confidence:", answer.confidence)
    print("Caveats:", "; ".join(answer.caveats))


if __name__ == "__main__":
    main()
