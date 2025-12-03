# HAVOC HYBRID Quick Start

**1 minute to first result** ⏱️

---

## Installation

```bash
# Already installed - just ensure src/ is in PYTHONPATH
cd /path/to/SLM
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in editable mode
pip install -e .
```

---

## Basic Usage

### One-Liner

```python
from havoc_prime import run_havoc_prime

result = run_havoc_prime("Your question here")
print(result["answer"])
```

### With Options

```python
from havoc_prime import HavocPrimeOrchestrator

orchestrator = HavocPrimeOrchestrator(
    enable_chrono=True,         # Iterative refinement
    enable_adversarial=True,    # Triple-fork reasoning
    max_chrono_iterations=3     # Max iterations
)

result = orchestrator.process("Your question")
print(result["answer"])
print(result["routing"])
print(result["verification"])
```

---

## Examples

### Trivial (MICRO)

```python
run_havoc_prime("3 + 5")
# Budget: MICRO, Time: ~10ms
```

### Simple (LIGHT)

```python
run_havoc_prime("What is a t-test?")
# Budget: LIGHT, Time: ~100ms
```

### Standard (MEDIUM)

```python
run_havoc_prime("Compare three groups using ANOVA")
# Budget: MEDIUM, Time: ~500ms
```

### Complex (HEAVY)

```python
run_havoc_prime("Design a Box-Behnken DOE for 3 factors")
# Budget: HEAVY, Time: ~1-2s
```

---

## Testing

```bash
# Run all tests
pytest tests/test_havoc_prime.py -v

# Run demo
python scripts/demo_havoc_prime.py

# Quick smoke test
python -c "from havoc_prime import run_havoc_prime; print(run_havoc_prime('3+3')['answer'])"
```

---

## Budget Levels

| Budget | When | Example |
|--------|------|---------|
| **MICRO** | Trivial | "3+3", "What is ANOVA?" |
| **LIGHT** | Simple | "Explain t-test" |
| **MEDIUM** | Standard | "Run ANOVA" |
| **HEAVY** | Complex | "Design DOE", "SPC analysis" |

---

## Result Structure

```python
result = {
    "answer": "...",                    # Formatted answer (string)
    "routing": {
        "task_type": "DOE",
        "budget": "HEAVY",
        "difficulty": "HARD",
        "risk": "HIGH"
    },
    "workspace_summary": {
        "facts_count": 10,
        "assumptions_count": 3,
        "constraints_count": 5,
        "global_confidence": 0.75
    },
    "verification": {
        "passed": True,
        "issues": [],
        "warnings": []
    },
    "formatted_answer_object": <Answer>  # For programmatic access
}
```

---

## SRS Tools (Direct Access)

```python
from srs_tools import PythonMathEngine, DSLExecutor, RAGHelper

# Math operations
math = PythonMathEngine()
ttest = math.t_test([1,2,3], [2,3,4])
anova = math.anova([[1,2], [3,4], [5,6]])

# DSL execution
dsl = DSLExecutor()
result = dsl.execute('{"DOE": {...}}')

# RAG retrieval
rag = RAGHelper()
refs = rag.retrieve("What is ANOVA?", k=5)
```

---

## Documentation

- **User Guide:** [`HAVOC_PRIME_README.md`](./HAVOC_PRIME_README.md)
- **Migration:** [`MIGRATION_GUIDE_SRS_TO_PRIME.md`](./MIGRATION_GUIDE_SRS_TO_PRIME.md)
- **Implementation:** [`HAVOC_HYBRID_IMPLEMENTATION_SUMMARY.md`](./HAVOC_HYBRID_IMPLEMENTATION_SUMMARY.md)

---

## Troubleshooting

### Import Error

```bash
# Fix: Add src/ to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Old SRS Pipeline

```python
# Don't use:
from havoc_srs.orchestrator import run_pipeline

# Use instead:
from havoc_prime import run_havoc_prime
```

---

## Next Steps

1. **Try the demos:**
   ```bash
   python scripts/demo_havoc_prime.py
   ```

2. **Read the docs:**
   - Start with `HAVOC_PRIME_README.md`

3. **Run tests:**
   ```bash
   pytest tests/test_havoc_prime.py -v
   ```

4. **Integrate with your code:**
   ```python
   from havoc_prime import run_havoc_prime
   result = run_havoc_prime("Your question")
   ```

---

**Ready to go!** 🚀

For questions: See `HAVOC_PRIME_README.md` → FAQ section
