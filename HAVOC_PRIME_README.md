# HAVOC PRIME - Hybrid Meta-Reasoning System

**Version:** 1.0.0
**Status:** Production-Ready Scaffold
**Date:** November 24, 2025

---

## What is HAVOC PRIME?

**HAVOC PRIME** is a hybrid AI reasoning system that combines:

1. **PRIME** = Meta-reasoning operating system (the "thinking" layer)
2. **SRS Tools** = Domain-specific toolbox (the "doing" layer)

Think of it as:
- **PRIME** = The architect who plans, reasons, debates, and verifies
- **SRS Tools** = The specialized workers (math, stats, DOE, SPC)

---

## Quick Start

### Basic Usage

```python
from havoc_prime import run_havoc_prime

# Simple question (MICRO budget - direct answer)
result = run_havoc_prime("3 + 5")
print(result["answer"])

# Complex task (HEAVY budget - full PRIME reasoning)
result = run_havoc_prime("Design a Box-Behnken DOE for temperature, pressure, and speed")
print(result["answer"])
print(result["routing"])        # See budget assignment
print(result["verification"])   # See verification report
```

### Custom Configuration

```python
from havoc_prime import HavocPrimeOrchestrator

orchestrator = HavocPrimeOrchestrator(
    enable_chrono=True,          # Enable iterative refinement
    enable_adversarial=True,     # Enable triple-fork reasoning
    max_chrono_iterations=3      # Max refinement iterations
)

result = orchestrator.process("Your question here")
```

---

## Budget Levels

PRIME adapts its reasoning depth based on task complexity:

| Budget | When | Subgoals | Adversarial | Chrono | Example |
|--------|------|----------|-------------|--------|---------|
| **MICRO** | Trivial tasks | 0 (direct) | No | No | "3 + 3", "What is ANOVA?" |
| **LIGHT** | Simple questions | 1-3 | No | No | "Explain t-test" |
| **MEDIUM** | Standard tasks | 3-7 | Yes | Yes | "Run ANOVA" |
| **HEAVY** | Complex analysis | 7-12 | Yes | Yes | "Design DOE", "SPC analysis" |

**Anti-Bloat Guarantee:** PRIME never wastes computation on trivial tasks.

---

## Architecture Overview

### PRIME Components

```
1. ROUTER → Classify task + assign budget
2. OPERATOR GRAPH → Decompose into subgoals (task-specific)
3. WORKSPACE → Shared memory (facts, assumptions, constraints)
4. CONSTRAINT BACKBONE → Enforce domain rules

For each subgoal:
5. CHRONO-LOOP → Iterative latent refinement
6. TRIPLE-FORK → Advocate vs HAVOC-ATTACK vs Pragmatist
7. CALL SRS TOOLS → If needed (math, DOE, SPC, RAG)
8. SYNTHESIS → Battle-tested local result

After all subgoals:
9. GLOBAL VERIFICATION → Check contradictions
10. GLOBAL HAVOC-ATTACK → Final adversarial check
11. FINAL COMPRESSION → Clean answer
12. ANSWER FORMATTER → Structured output
```

### SRS Tools (Callable Modules)

```python
from srs_tools import PythonMathEngine, DSLExecutor, RAGHelper

# Math/stats operations
math = PythonMathEngine()
ttest_result = math.t_test([1,2,3], [2,3,4])
anova_result = math.anova([[1,2], [3,4], [5,6]])

# DSL execution (DOE/SPC/STATS)
dsl = DSLExecutor()
result = dsl.execute('{"DOE": {...}}')

# RAG retrieval
rag = RAGHelper()
references = rag.retrieve("What is ANOVA?", k=5)
```

---

## Key Features

### 1. Dynamic Subgoal Graphs

Unlike the old SRS pipeline (fixed 8 stages), PRIME builds task-specific subgoals:

**DOE Task:**
```
doe_1: Identify factors and levels
  └─> doe_2: Select design type
        └─> doe_3: Generate design matrix
              └─> doe_4: Validate design properties
```

**SPC Task:**
```
spc_1: Identify chart type
  └─> spc_2: Calculate control limits
        └─> spc_3: Apply Western Electric rules
```

### 2. Triple-Fork Adversarial Reasoning

Every subgoal result is battle-tested by three perspectives:

1. **Advocate** (Optimistic)
   - "p-value < 0.05 → statistically significant!"
   - "R² = 0.85 → excellent fit!"

2. **HAVOC-ATTACK** (Critical)
   - "p-value = 0.04 → barely significant, could be Type I error"
   - "Small sample size (n=20) → unreliable"

3. **Pragmatist** (Practical)
   - "Statistically significant but effect size is small"
   - "Results are sound but assumptions need validation"

**Synthesis:** Weighted combination → battle-tested conclusion

### 3. Chrono-Loop (Iterative Refinement)

Subgoal results are refined through multiple latent passes:

```
Iteration 1: Initial result (confidence: 0.5)
Iteration 2: Stability-tested (confidence: 0.65)
Iteration 3: Converged (confidence: 0.7)
```

Stops when:
- Convergence reached (diff < threshold)
- Max iterations hit
- Stability confirmed

### 4. Global Verification

After all subgoals complete, PRIME checks:
- ✅ Constraint violations
- ✅ Failed subgoals
- ✅ Contradictory facts
- ✅ Low confidence facts
- ✅ Critical assumptions

Violations → confidence penalty + warnings in final answer.

### 5. Constraint Backbone

Domain-specific constraints enforced throughout:

**STATS:** Sample size adequacy, assumption validation, effect size reporting
**DOE:** Design balance, confounding check, factor independence
**SPC:** Baseline stability, subgroup rationality, measurement accuracy

Violations are tracked and reported.

---

## Examples

### Example 1: Trivial Arithmetic (MICRO)

```python
result = run_havoc_prime("3 + 5")

# Output:
# Budget: MICRO (no PRIME reasoning)
# Answer: "Result: 8"
# Confidence: 100%
```

### Example 2: Stats Question (MEDIUM)

```python
result = run_havoc_prime("Compare means of three groups using ANOVA")

# Output:
# Budget: MEDIUM
# Subgoals: 3 (identify test, execute, interpret)
# Adversarial: Yes (Advocate/Attack/Pragmatist)
# Verification: Passed
# Answer: [Full structured answer with key numbers, caveats, checks]
```

### Example 3: DOE Design (HEAVY)

```python
result = run_havoc_prime("Design a Box-Behnken DOE for 3 factors")

# Output:
# Budget: HEAVY
# Subgoals: 4 (identify factors, select design, generate matrix, validate)
# Adversarial: Yes (triple-fork per subgoal + global)
# Chrono-Loop: Yes (2-3 iterations per subgoal)
# SRS Tools Called: dsl_executor, python_math
# Verification: Passed (no critical violations)
# Answer: [Comprehensive structured answer with design matrix, assumptions, checks]
```

---

## Testing

### Run All Tests

```bash
pytest tests/test_havoc_prime.py -v
```

### Run Demo

```bash
python scripts/demo_havoc_prime.py
```

This will demonstrate MICRO, LIGHT, MEDIUM, and HEAVY budgets.

---

## Migration from SRS-7B

See [`MIGRATION_GUIDE_SRS_TO_PRIME.md`](./MIGRATION_GUIDE_SRS_TO_PRIME.md) for detailed migration instructions.

**TL;DR:**

```python
# OLD (SRS-7B pipeline)
from havoc_srs.orchestrator import run_pipeline
answer = run_pipeline("question", config)

# NEW (HAVOC PRIME)
from havoc_prime import run_havoc_prime
result = run_havoc_prime("question")
```

---

## File Structure

```
src/
├── havoc_prime/              # PRIME meta-reasoning
│   ├── router.py             # Task classification + budget
│   ├── operator_graph.py     # Subgoal decomposition
│   ├── workspace.py          # Global memory
│   ├── constraints.py        # Constraint enforcement
│   ├── adversarial.py        # Triple-fork reasoning
│   ├── chrono_loop.py        # Latent refinement
│   ├── verification.py       # Global checks
│   ├── compression.py        # Answer compression
│   └── orchestrator.py       # Main coordinator
│
├── srs_tools/                # SRS domain tools
│   ├── dsl_executor.py       # DSL execution
│   ├── python_math.py        # Math/stats operations
│   ├── rag_helper.py         # RAG retrieval
│   └── answer_formatter.py   # Answer formatting
│
└── havoc_srs/                # OLD pipeline (kept for reference)
```

---

## Performance

### Latency Estimates

| Budget | Subgoals | Latency (est.) |
|--------|----------|----------------|
| MICRO | 0 | ~10ms |
| LIGHT | 1-3 | ~100ms |
| MEDIUM | 3-7 | ~500ms |
| HEAVY | 7-12 | ~1-2s |

**Note:** These are estimates for the current scaffold. Actual latency will depend on model inference speed once integrated.

### Anti-Bloat Measures

1. ✅ MICRO budget skips all reasoning
2. ✅ Subgoal limits (LIGHT: 3, MEDIUM: 7, HEAVY: 12)
3. ✅ Chrono-loop capped (max 2-3 iterations)
4. ✅ Adversarial only when budget allows
5. ✅ SRS tools called max 1-3x per subgoal
6. ✅ No macro-loop unless critical failure

---

## Roadmap

### Phase 1: Scaffold (DONE ✅)
- ✅ PRIME architecture
- ✅ Budget control
- ✅ Adversarial reasoning
- ✅ SRS tool refactor
- ✅ Tests and demos

### Phase 2: Model Integration (TODO)
- ⏳ Connect to HAVOC-7B model
- ⏳ Real latent chrono-loop (using hidden states)
- ⏳ Chaos injection (dropout testing)
- ⏳ Vector stream (compressed latent memory)
- ⏳ Macro-loop (full reasoning restart)

### Phase 3: Production (TODO)
- ⏳ Multi-GPU support
- ⏳ Distributed PRIME
- ⏳ Production configs
- ⏳ Benchmark suite
- ⏳ Web UI with budget selector

---

## FAQ

### Q: Is this production-ready?

**A:** The architecture is production-ready, but model integration is needed. Current version uses:
- ✅ Functional PRIME reasoning scaffold
- ✅ Working SRS tools
- ⚠️ Simulated chrono-loop (needs model hidden states)
- ⚠️ Placeholder chaos injection

### Q: Can I use the old SRS pipeline?

**A:** Yes! The old `havoc_srs/` pipeline is kept for backward compatibility:

```python
from havoc_srs.orchestrator import run_pipeline
answer = run_pipeline("question", config)
```

But we recommend migrating to PRIME.

### Q: How do I control budget?

**A:** Budget is auto-assigned by the router based on task complexity. To force a budget:

```python
from havoc_prime import TaskRouter, HavocPrimeOrchestrator, Budget

# Manual budget assignment (for testing)
router = TaskRouter()
decision = router.route("your question")
decision.budget = Budget.HEAVY  # Force HEAVY

# Then use custom orchestrator...
```

### Q: Can I disable adversarial reasoning?

**A:** Yes:

```python
orchestrator = HavocPrimeOrchestrator(
    enable_adversarial=False,
    enable_chrono=False
)
```

This gives you basic PRIME without adversarial/chrono overhead.

### Q: How do I add new SRS tools?

**A:** Create a new module in `srs_tools/`:

```python
# srs_tools/my_new_tool.py
class MyNewTool:
    def run(self, params):
        # Your tool logic
        return result
```

Then call it in orchestrator's `_execute_subgoal` method.

---

## License

Apache 2.0 (same as HAVOC-7B)

---

## Citation

If you use HAVOC PRIME in research:

```bibtex
@software{havoc_prime_2025,
  title={HAVOC PRIME: Hybrid Meta-Reasoning for Domain-Specialized AI},
  author={HAVOC Team},
  year={2025},
  url={https://github.com/your-repo/havoc}
}
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Docs:** See `MIGRATION_GUIDE_SRS_TO_PRIME.md`, `COMPREHENSIVE_REPO_ANALYSIS.md`
- **Examples:** `scripts/demo_havoc_prime.py`

---

**Built with:** PyTorch, NumPy, SciPy, statsmodels
**Tested on:** Python 3.10, 3.11, 3.12
**License:** Apache 2.0

---

**Last Updated:** November 24, 2025
