# Migration Guide: SRS-7B → HAVOC HYBRID (PRIME + SRS Tools)

**Date:** November 24, 2025
**Version:** 1.0.0

---

## Overview

This guide documents the migration from the **SRS-7B pipeline** (8-stage linear reasoning) to the **HAVOC HYBRID system** (PRIME meta-reasoning + SRS tools).

### What Changed

| Component | Old (SRS-7B) | New (HAVOC HYBRID) |
|-----------|-------------|-------------------|
| **Architecture** | Linear 8-stage pipeline | PRIME meta-reasoning + SRS toolbox |
| **Reasoning** | Hard-coded stages | Dynamic subgoal graph + adversarial reasoning |
| **Tools** | Embedded in pipeline | Isolated callable modules |
| **Budget Control** | None | MICRO/LIGHT/MEDIUM/HEAVY |
| **Adversarial** | Simple PRO/CON | Advocate/HAVOC-ATTACK/Pragmatist |
| **Verification** | Single audit stage | Global verification + constraint backbone |

---

## Architecture Comparison

### Old: SRS-7B Pipeline

```
MODE → GROUND → PLAN → EXECUTE → ARGUE → ARBITER → AUDIT → ANSWER
```

**Problems:**
- ❌ Fixed flow - can't skip stages or adapt
- ❌ All tasks get same treatment (no budget control)
- ❌ Tools embedded in EXECUTE stage
- ❌ Simple PRO/CON reasoning
- ❌ No iterative refinement

### New: HAVOC HYBRID

```
ROUTE → (MICRO: direct) OR (PRIME: dynamic reasoning + SRS tools)

PRIME Flow:
1. Build operator graph (task-specific subgoals)
2. For each subgoal:
   - Chrono-loop (latent refinement)
   - Triple-fork (Advocate/Attack/Pragmatist)
   - Call SRS tools if needed
   - Synthesize local result
3. Global verification
4. Final compression
5. Format answer
```

**Benefits:**
- ✅ Adaptive flow based on budget
- ✅ Budget control (MICRO/LIGHT/MEDIUM/HEAVY)
- ✅ SRS tools callable from anywhere
- ✅ Adversarial battle-testing
- ✅ Iterative refinement via chrono-loop

---

## File Structure Changes

### New Directories

```
src/
├── havoc_prime/              # NEW: PRIME meta-reasoning
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
├── srs_tools/                # NEW: Refactored SRS tools
│   ├── dsl_executor.py       # DSL execution (DOE/SPC/STATS)
│   ├── python_math.py        # Math/stats operations
│   ├── rag_helper.py         # RAG retrieval
│   └── answer_formatter.py   # Answer formatting
│
└── havoc_srs/                # OLD: Keep for reference (archive)
    └── [old pipeline modules]
```

### Deprecated Files

These files are **kept for reference** but no longer used:

- `havoc_srs/orchestrator.py` - Replaced by `havoc_prime/orchestrator.py`
- `havoc_srs/mode.py` - Replaced by `havoc_prime/router.py`
- `havoc_srs/plan.py` - Replaced by `havoc_prime/operator_graph.py`
- `havoc_srs/execute.py` - Logic moved to `srs_tools/*`
- `havoc_srs/argue.py` - Replaced by `havoc_prime/adversarial.py`
- `havoc_srs/arbiter.py` - Replaced by adversarial synthesis
- `havoc_srs/audit.py` - Replaced by `havoc_prime/verification.py`
- `havoc_srs/answer.py` - Replaced by `srs_tools/answer_formatter.py`

---

## Code Migration

### Old Usage (SRS-7B)

```python
from havoc_core.config import SRSConfig
from havoc_srs.orchestrator import run_pipeline

# Old way
answer = run_pipeline(prompt="Design a DOE", config=SRSConfig())
print(answer)
```

### New Usage (HAVOC HYBRID)

```python
from havoc_prime.orchestrator import run_havoc_prime

# New way
result = run_havoc_prime("Design a DOE")

print(result["answer"])            # Formatted answer
print(result["routing"])            # Routing decision
print(result["workspace_summary"])  # Workspace state
print(result["verification"])       # Verification report
```

### Custom Orchestrator

```python
from havoc_prime.orchestrator import HavocPrimeOrchestrator

# Create custom orchestrator
orchestrator = HavocPrimeOrchestrator(
    enable_chrono=True,         # Enable chrono-loop
    enable_adversarial=True,    # Enable triple-fork
    max_chrono_iterations=3     # Max iterations
)

# Process prompt
result = orchestrator.process("Your question here")
```

### Calling SRS Tools Directly

```python
from srs_tools import PythonMathEngine, DSLExecutor, RAGHelper

# Math tool
math_engine = PythonMathEngine()
ttest_result = math_engine.t_test([1, 2, 3], [2, 3, 4])

# DSL tool
dsl_exec = DSLExecutor()
dsl_result = dsl_exec.execute('{"DOE": {...}}')

# RAG tool
rag = RAGHelper()
references = rag.retrieve("What is ANOVA?", k=5)
```

---

## Behavioral Changes

### Budget Assignment

**Old:** All tasks received full 8-stage pipeline treatment.

**New:** Tasks are assigned budgets based on complexity:

| Budget | When Used | PRIME Features |
|--------|-----------|----------------|
| **MICRO** | Trivial (3+3, "What is...") | None - direct answer |
| **LIGHT** | Simple questions | Minimal subgoals, no adversarial/chrono |
| **MEDIUM** | Standard tasks | Full PRIME, limited loops |
| **HEAVY** | Complex (DOE, SPC) | Full PRIME + adversarial + chrono |

### Reasoning Depth

**Old:** Fixed 8 stages for all tasks.

**New:** Dynamic subgoals based on task:
- LIGHT: 1-3 subgoals
- MEDIUM: 3-7 subgoals
- HEAVY: 7-12 subgoals

### Adversarial Reasoning

**Old:** Simple PRO/CON arguments, select winner.

**New:** Triple-fork reasoning:
1. **Advocate** - Best-case argument
2. **HAVOC-ATTACK** - Aggressive critic
3. **Pragmatist** - Practical feasibility
4. **Synthesis** - Battle-tested conclusion

### Confidence Tracking

**Old:** Simple confidence score with audit downgrade.

**New:** Multi-stage confidence evolution:
1. Initial confidence per subgoal
2. Chrono-loop stability boost
3. Adversarial synthesis adjustment
4. Constraint violation penalties
5. Verification report adjustment

---

## Testing

### Run Tests

```bash
# Test HAVOC PRIME
pytest tests/test_havoc_prime.py -v

# Test SRS tools
pytest tests/test_srs_tools.py -v
```

### Run Demos

```bash
# Demo all budget levels
python scripts/demo_havoc_prime.py

# Demo specific budget
python -c "from havoc_prime.orchestrator import run_havoc_prime; print(run_havoc_prime('3+3')['answer'])"
```

---

## Configuration

### Old Configuration (SRS)

```yaml
# configs/srs/default_srs.yaml
rag:
  embed_dim: 384
  top_k: 5

reasoning:
  enable_argue: true
  enable_arbiter: true
  enable_audit: true
```

### New Configuration (PRIME)

```python
# In code (no YAML needed for basic usage)
orchestrator = HavocPrimeOrchestrator(
    enable_chrono=True,
    enable_adversarial=True,
    max_chrono_iterations=3
)
```

---

## Performance Considerations

### Anti-Bloat Rules

PRIME enforces strict limits to prevent computational explosion:

1. ✅ **MICRO budget skips PRIME entirely**
2. ✅ **Subgoal limits** (LIGHT: 3, MEDIUM: 7, HEAVY: 12)
3. ✅ **Chrono-loop capped** (max 2-3 iterations)
4. ✅ **Adversarial only when needed** (MEDIUM+)
5. ✅ **SRS tools called max 1-3x per subgoal**

### Expected Latency

| Budget | Subgoals | Adversarial | Chrono | Latency (est.) |
|--------|----------|-------------|--------|----------------|
| MICRO | 0 | No | No | ~10ms |
| LIGHT | 1-3 | No | No | ~100ms |
| MEDIUM | 3-7 | Yes | Yes | ~500ms |
| HEAVY | 7-12 | Yes | Yes | ~1-2s |

---

## Troubleshooting

### Issue: "Module not found: havoc_prime"

**Solution:** Ensure you're in the project root and Python can find `src/`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or install in editable mode:

```bash
pip install -e .
```

### Issue: "Module not found: srs_tools"

**Solution:** Same as above - ensure `src/` is in PYTHONPATH.

### Issue: Tests fail with import errors

**Solution:** Run from project root:

```bash
cd /path/to/SLM
pytest tests/test_havoc_prime.py -v
```

### Issue: Old SRS pipeline still being called

**Solution:** Update imports:

```python
# OLD - don't use
from havoc_srs.orchestrator import run_pipeline

# NEW - use this
from havoc_prime.orchestrator import run_havoc_prime
```

---

## Backward Compatibility

### Old SRS Pipeline

The old SRS pipeline is **still available** in `havoc_srs/` for backward compatibility.

To use old pipeline:

```python
from havoc_srs.orchestrator import run_pipeline
from havoc_core.config import SRSConfig

answer = run_pipeline("Your question", SRSConfig())
```

**Note:** Old pipeline will not receive updates. Migrate to PRIME for new features.

---

## Roadmap

### Phase 1 (DONE)
- ✅ PRIME architecture
- ✅ SRS tool refactor
- ✅ Budget control
- ✅ Adversarial reasoning
- ✅ Tests and demos

### Phase 2 (Future)
- ⏳ Model integration (actual latent chrono-loop)
- ⏳ Macro-loop (full reasoning restart)
- ⏳ Chaos injection (dropout testing)
- ⏳ Vector stream (latent memory)
- ⏳ Web UI with budget selector

### Phase 3 (Future)
- ⏳ Multi-model support
- ⏳ Distributed PRIME (multi-GPU)
- ⏳ Production deployment configs
- ⏳ Benchmark suite

---

## Support

**Issues:** https://github.com/anthropics/claude-code/issues
**Documentation:** See `README.md`, `COMPREHENSIVE_REPO_ANALYSIS.md`

---

**Last Updated:** November 24, 2025
