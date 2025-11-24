from havoc_core.config import SRSConfig
from havoc_srs.orchestrator import run_pipeline


def main() -> None:
    prompt = "Design a Box-Behnken DOE for temperature, speed, and pressure"
    answer = run_pipeline(prompt, SRSConfig())
    print("Answer:", answer)


if __name__ == "__main__":
    main()
