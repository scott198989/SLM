"""
HAVOC PRIME Demo Script

Demonstrates MICRO, LIGHT, MEDIUM, and HEAVY budget execution.
"""

from havoc_prime.orchestrator import run_havoc_prime


def demo_micro_budget():
    """Demo MICRO budget (no PRIME reasoning)"""
    print("=" * 80)
    print("MICRO BUDGET DEMO - Trivial Arithmetic")
    print("=" * 80)

    result = run_havoc_prime("3 + 5")

    print(f"\nRouting: {result['routing']}")
    print(f"\n{result['answer']}")
    print()


def demo_light_budget():
    """Demo LIGHT budget (minimal PRIME)"""
    print("=" * 80)
    print("LIGHT BUDGET DEMO - Simple Question")
    print("=" * 80)

    result = run_havoc_prime("What is a t-test?")

    print(f"\nRouting: {result['routing']}")
    print(f"\nWorkspace: {result['workspace_summary']}")
    print(f"\n{result['answer']}")
    print()


def demo_medium_budget():
    """Demo MEDIUM budget (standard PRIME + limited adversarial)"""
    print("=" * 80)
    print("MEDIUM BUDGET DEMO - Stats Analysis")
    print("=" * 80)

    result = run_havoc_prime("Compare three groups using ANOVA")

    print(f"\nRouting: {result['routing']}")
    print(f"\nWorkspace: {result['workspace_summary']}")
    print(f"\nVerification: {result['verification']}")
    print(f"\n{result['answer']}")
    print()


def demo_heavy_budget():
    """Demo HEAVY budget (full PRIME + SRS tools)"""
    print("=" * 80)
    print("HEAVY BUDGET DEMO - DOE Design")
    print("=" * 80)

    result = run_havoc_prime("Design a Box-Behnken DOE for temperature, pressure, and speed")

    print(f"\nRouting: {result['routing']}")
    print(f"\nWorkspace: {result['workspace_summary']}")
    print(f"\nVerification: {result['verification']}")
    print(f"\n{result['answer']}")
    print()


def demo_spc_analysis():
    """Demo SPC analysis with HEAVY budget"""
    print("=" * 80)
    print("HEAVY BUDGET DEMO - SPC Analysis")
    print("=" * 80)

    result = run_havoc_prime("Analyze control chart data for out-of-control signals")

    print(f"\nRouting: {result['routing']}")
    print(f"\nWorkspace: {result['workspace_summary']}")
    print(f"\nVerification: {result['verification']}")
    print(f"\n{result['answer']}")
    print()


def main():
    """Run all demos"""
    print("\n\n")
    print("=" * 80)
    print(" HAVOC PRIME DEMONSTRATION")
    print(" Hybrid Reasoning: PRIME (meta-reasoning) + SRS (domain tools)")
    print("=" * 80)
    print("\n\n")

    # Demo each budget level
    demo_micro_budget()
    demo_light_budget()
    demo_medium_budget()
    demo_heavy_budget()
    demo_spc_analysis()

    print("=" * 80)
    print("DEMOS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
