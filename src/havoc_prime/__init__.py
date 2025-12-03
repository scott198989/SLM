"""
HAVOC PRIME - Meta-Reasoning Operating System

Hybrid AI system combining:
- PRIME: Meta-reasoning architecture (dynamic subgoals, adversarial reasoning, verification)
- SRS Tools: Domain-specific toolbox (DOE/SPC/STATS)

Quick Start:
    from havoc_prime import run_havoc_prime

    result = run_havoc_prime("Design a Box-Behnken DOE")
    print(result["answer"])

Components:
    - router: Task classification + budget assignment
    - operator_graph: Dynamic subgoal decomposition
    - workspace: Global shared memory
    - constraints: Domain constraint enforcement
    - adversarial: Triple-fork reasoning (Advocate/Attack/Pragmatist)
    - chrono_loop: Latent iterative refinement
    - verification: Global consistency checking
    - compression: Answer compression
    - orchestrator: Main coordinator

Budget Levels:
    - MICRO: Direct answer (no reasoning)
    - LIGHT: Minimal PRIME (1-3 subgoals)
    - MEDIUM: Standard PRIME (3-7 subgoals, adversarial)
    - HEAVY: Full PRIME (7-12 subgoals, adversarial, chrono-loop)
"""

from havoc_prime.adversarial import (
    Advocate,
    AdversarialSynthesizer,
    Argument,
    HavocAttack,
    Pragmatist,
    Synthesis,
)
from havoc_prime.chrono_loop import ChronoLoop, ChronoState
from havoc_prime.compression import FinalCompression
from havoc_prime.constraints import ConstraintBackbone
from havoc_prime.operator_graph import (
    OperatorGraph,
    OperatorGraphBuilder,
    Subgoal,
)
from havoc_prime.orchestrator import HavocPrimeOrchestrator, run_havoc_prime
from havoc_prime.router import (
    Budget,
    Difficulty,
    Risk,
    RoutingDecision,
    TaskRouter,
    TaskType,
)
from havoc_prime.verification import GlobalVerification, VerificationReport
from havoc_prime.workspace import (
    Assumption,
    Constraint,
    Fact,
    GlobalWorkspace,
)

__version__ = "1.0.0"

__all__ = [
    # Main entry point
    "run_havoc_prime",
    "HavocPrimeOrchestrator",

    # Router
    "TaskRouter",
    "RoutingDecision",
    "Budget",
    "TaskType",
    "Difficulty",
    "Risk",

    # Operator Graph
    "OperatorGraphBuilder",
    "OperatorGraph",
    "Subgoal",

    # Workspace
    "GlobalWorkspace",
    "Fact",
    "Assumption",
    "Constraint",

    # Constraints
    "ConstraintBackbone",

    # Adversarial
    "Advocate",
    "HavocAttack",
    "Pragmatist",
    "AdversarialSynthesizer",
    "Argument",
    "Synthesis",

    # Chrono Loop
    "ChronoLoop",
    "ChronoState",

    # Verification
    "GlobalVerification",
    "VerificationReport",

    # Compression
    "FinalCompression",
]
