# CLAUDE.md - AI Assistant Guide for HAVOC-7B/SIGMA-7B

**Last Updated:** November 24, 2025
**Project:** HAVOC-7B (SIGMA-7B) - Domain-Specialized 7B Transformer with SRS Reasoning Stack
**License:** Apache 2.0

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Project Architecture](#project-architecture)
3. [Development Workflows](#development-workflows)
4. [Code Conventions](#code-conventions)
5. [Key Files and Locations](#key-files-and-locations)
6. [Common Tasks](#common-tasks)
7. [Testing and Validation](#testing-and-validation)
8. [Deployment Considerations](#deployment-considerations)
9. [Important Context](#important-context)

---

## Repository Overview

### What This Project Is

HAVOC-7B (also called SIGMA-7B) is a **from-scratch 7 billion parameter decoder-only transformer** specialized for mathematics, statistics, engineering, and manufacturing intelligence (DOE/SPC). This is a complete ML system including:

- ✅ Custom 7B transformer model (GQA, RoPE, SwiGLU, RMSNorm)
- ✅ Production training pipeline (mixed precision, checkpointing, validation)
- ✅ FastAPI inference server (streaming, multiple sampling strategies)
- ✅ React chat UI with 3D graphics (Three.js)
- ✅ Scott Reasoning Stack (SRS-7B) - 8-stage reasoning pipeline
- ✅ Math/stats tools and DSL for DOE/SPC
- ✅ RAG layer scaffolding
- ✅ Comprehensive documentation (2000+ lines)

**Total codebase:** ~3,500 lines of Python + ~1,100 lines of JavaScript/React

### What This Project Is NOT

- ❌ NOT a pre-trained model (no weights included)
- ❌ NOT using Hugging Face Transformers library
- ❌ NOT multi-GPU/distributed (single GPU only)
- ❌ NOT production-deployed (local development/training)

### Project Status

**Production-ready components:**
- Model architecture ✅
- Training pipeline ✅
- Inference server ✅
- Chat UI ✅
- Documentation ✅

**Needs training/data:**
- Model weights (random initialization currently)
- Tokenizer model (needs corpus)
- RAG index (needs documents)
- Evaluation benchmarks (needs datasets)

---

## Project Architecture

### Directory Structure

```
SLM/
├── src/                          # Python source code (~2,340 lines)
│   ├── havoc_core/               # Core model & config (7 files, ~500 lines)
│   │   ├── config.py             # All configuration dataclasses
│   │   ├── model/
│   │   │   ├── transformer.py   # Main HavocModel class
│   │   │   └── blocks.py        # RMSNorm, GQA, RoPE, SwiGLU
│   │   └── tokenizer/
│   │       ├── train_tokenizer.py   # SentencePiece training
│   │       └── vocab_utils.py       # Special tokens
│   │
│   ├── havoc_data/               # Data pipeline (4 files, ~150 lines)
│   │   ├── sources.py            # DataSource management
│   │   ├── preprocess.py         # Text normalization
│   │   └── dataset.py            # PyTorch Dataset
│   │
│   ├── havoc_training/           # Training infrastructure (2 files, ~580 lines)
│   │   └── trainer.py            # Trainer class (AdamW, AMP, checkpointing)
│   │
│   ├── havoc_inference/          # Inference server (3 files, ~490 lines)
│   │   ├── engine.py             # InferenceEngine (generation, sampling)
│   │   └── server.py             # FastAPI application
│   │
│   ├── havoc_tools/              # Math/stats tools (6 files, ~200 lines)
│   │   ├── python_math/engine.py # T-tests, ANOVA, regression, DOE
│   │   └── dsl/                  # DSL for DOE/SPC
│   │
│   ├── havoc_rag/                # RAG layer (4 files, ~150 lines)
│   │   ├── embeddings.py         # Embedding wrapper
│   │   ├── index.py              # Vector index
│   │   └── retrieval.py          # Retrieval interface
│   │
│   ├── havoc_srs/                # Scott Reasoning Stack (10 files, ~400 lines)
│   │   ├── orchestrator.py       # 8-stage pipeline
│   │   └── [mode|ground|plan|execute|argue|arbiter|audit|answer].py
│   │
│   ├── havoc_eval/               # Evaluation (3 files, ~100 lines)
│   └── havoc_cli/                # CLI (2 files, ~50 lines)
│
├── frontend/                     # React chat UI (~1,069 lines)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Scene3D.jsx       # Three.js 3D background
│   │   │   ├── ChatMessage.jsx   # Message components
│   │   │   ├── ChatInput.jsx     # Input area
│   │   │   ├── MarkdownMessage.jsx  # Markdown/code/math rendering
│   │   │   └── Settings.jsx      # Settings modal
│   │   ├── lib/api.js            # API client
│   │   ├── App.jsx               # Main app
│   │   └── index.css             # Tailwind styles
│   └── [vite.config.js, tailwind.config.js, package.json]
│
├── configs/                      # YAML configurations (7 files)
│   ├── model/havoc_7b.yaml       # Model architecture
│   ├── training/default_training.yaml
│   ├── inference/default_inference.yaml
│   └── [data|rag|srs|tools]/
│
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training CLI (280 lines)
│   ├── serve.py                  # Inference server CLI (140 lines)
│   └── demo_run.py               # SRS demo
│
├── tests/                        # Unit tests
├── README.md                     # Main README
├── TRAINING_GUIDE.md             # Training documentation
├── INFERENCE_GUIDE.md            # API documentation
├── COMPREHENSIVE_REPO_ANALYSIS.md # Full analysis report
└── pyproject.toml                # Package configuration
```

### Core Components

#### 1. **Model Architecture** (`src/havoc_core/`)

**Key file:** `src/havoc_core/model/transformer.py:13`

```python
class HavocModel(nn.Module):
    """7B parameter decoder-only transformer"""
    - 32 layers
    - d_model: 4096
    - 32 attention heads (8 KV heads for GQA)
    - SwiGLU MLP (~11k hidden)
    - RoPE positional embeddings
    - RMSNorm
    - KV-cache support
```

**Important:** Model alias `SigmaModel = HavocModel` exists for naming flexibility.

#### 2. **Training Pipeline** (`src/havoc_training/`)

**Key file:** `src/havoc_training/trainer.py:1`

The `Trainer` class handles:
- AdamW optimizer (excludes bias/norm from weight decay)
- Cosine LR schedule with warmup
- Mixed precision (AMP with bfloat16/float16)
- Gradient accumulation & clipping
- Checkpoint save/load/resume
- Validation with perplexity
- Comprehensive logging

**Entry point:** `scripts/train.py`

#### 3. **Inference Server** (`src/havoc_inference/`)

**Key file:** `src/havoc_inference/server.py:1`

FastAPI application with endpoints:
- `POST /completion` - Text completion
- `POST /chat` - Chat with message history
- Both support streaming (Server-Sent Events)

**Entry point:** `scripts/serve.py`

#### 4. **Frontend** (`frontend/`)

**Key file:** `frontend/src/App.jsx:1`

React 18 + Vite application with:
- Three.js 3D background (`Scene3D.jsx`)
- Markdown/LaTeX/code rendering (`MarkdownMessage.jsx`)
- Settings panel with presets (`Settings.jsx`)
- API client with streaming support (`lib/api.js`)

#### 5. **SRS Reasoning Stack** (`src/havoc_srs/`)

**Key file:** `src/havoc_srs/orchestrator.py:1`

8-stage reasoning pipeline:
```
MODE → GROUND → PLAN → EXECUTE → ARGUE → ARBITER → AUDIT → ANSWER
```

Each stage is a separate module with specific responsibilities.

---

## Development Workflows

### Workflow 1: Model Development

**When modifying the model architecture:**

1. **Update config first:** Edit `src/havoc_core/config.py`
2. **Modify architecture:** Edit `src/havoc_core/model/blocks.py` or `transformer.py`
3. **Update tests:** Add/modify tests in `tests/test_config.py`
4. **Verify training:** Run `python scripts/train.py --max-steps 10`
5. **Check inference:** Test with `python scripts/serve.py`

**Files to touch:**
- `src/havoc_core/config.py` - Configuration
- `src/havoc_core/model/*.py` - Architecture
- `configs/model/havoc_7b.yaml` - YAML config
- `tests/test_*.py` - Tests

### Workflow 2: Training Changes

**When modifying training logic:**

1. **Update TrainingConfig:** Edit `src/havoc_core/config.py:TrainingConfig`
2. **Modify Trainer:** Edit `src/havoc_training/trainer.py`
3. **Update default config:** Edit `configs/training/default_training.yaml`
4. **Update documentation:** Edit `TRAINING_GUIDE.md`
5. **Test training:** Run with dummy data to verify

**Files to touch:**
- `src/havoc_training/trainer.py`
- `src/havoc_core/config.py`
- `configs/training/default_training.yaml`
- `TRAINING_GUIDE.md`

### Workflow 3: Inference Changes

**When modifying inference/serving:**

1. **Update InferenceConfig:** Edit `src/havoc_core/config.py:InferenceConfig`
2. **Modify engine:** Edit `src/havoc_inference/engine.py`
3. **Update server:** Edit `src/havoc_inference/server.py` (if API changes)
4. **Update frontend client:** Edit `frontend/src/lib/api.js` (if API changes)
5. **Update documentation:** Edit `INFERENCE_GUIDE.md`

**Files to touch:**
- `src/havoc_inference/engine.py`
- `src/havoc_inference/server.py`
- `frontend/src/lib/api.js`
- `INFERENCE_GUIDE.md`

### Workflow 4: Frontend Changes

**When modifying the UI:**

1. **Edit components:** Modify `frontend/src/components/*.jsx`
2. **Update styles:** Edit `frontend/src/index.css` or Tailwind classes
3. **Test locally:** Run `npm run dev`
4. **Build:** Run `npm run build` to verify
5. **Update docs:** Edit `frontend/README.md` if needed

**Files to touch:**
- `frontend/src/components/*.jsx`
- `frontend/src/App.jsx`
- `frontend/src/index.css`
- `frontend/README.md`

### Workflow 5: Adding New Tools

**When adding math/stats tools:**

1. **Add function:** Edit `src/havoc_tools/python_math/engine.py`
2. **Add to DSL:** Edit `src/havoc_tools/dsl/spec.py` (if DSL needed)
3. **Update executor:** Edit `src/havoc_tools/dsl/executor.py`
4. **Add config:** Edit `configs/tools/default_tools.yaml`
5. **Add tests:** Create test in `tests/`

**Files to touch:**
- `src/havoc_tools/python_math/engine.py`
- `src/havoc_tools/dsl/*.py`
- `configs/tools/default_tools.yaml`

---

## Code Conventions

### Python Style

**General conventions:**
- **Style:** Black formatter (line length 100)
- **Imports:** isort with black profile
- **Type hints:** Use throughout (from `__future__ import annotations`)
- **Docstrings:** Optional for small functions, required for public APIs
- **Config:** Dataclasses with defaults (see `src/havoc_core/config.py`)

**Naming:**
- Classes: `PascalCase` (e.g., `HavocModel`, `Trainer`)
- Functions/methods: `snake_case` (e.g., `train_epoch`, `generate_text`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_VOCAB_SIZE`)
- Private: Prefix with `_` (e.g., `_build_attention_mask`)

**Configuration pattern:**
```python
from dataclasses import dataclass, field

@dataclass
class MyConfig:
    """Configuration for MyComponent."""
    param1: int = 100
    param2: float = 0.1
    param3: list = field(default_factory=list)
```

**Model pattern:**
```python
class MyModel(nn.Module):
    def __init__(self, config: MyConfig):
        super().__init__()
        self.config = config
        # Initialize layers

    @classmethod
    def from_config(cls, config: MyConfig) -> "MyModel":
        return cls(config)
```

### JavaScript/React Style

**General conventions:**
- **Style:** ESLint + Prettier
- **Components:** Functional components with hooks
- **Props:** Destructure in function signature
- **Styling:** Tailwind utility classes (avoid inline styles)

**Component pattern:**
```jsx
export function MyComponent({ prop1, prop2, onEvent }) {
  const [state, setState] = useState(initialValue);

  useEffect(() => {
    // Side effects
  }, [dependencies]);

  return (
    <div className="tailwind classes">
      {/* Content */}
    </div>
  );
}
```

**API client pattern:**
```javascript
// Use async/await
export async function myApiCall(params) {
  const response = await fetch(url, { /* options */ });
  if (!response.ok) {
    throw new Error(`Failed: ${response.status}`);
  }
  return await response.json();
}

// For streaming, use async generators
export async function* streamingCall(params) {
  const response = await fetch(url, { /* options */ });
  const reader = response.body.getReader();
  // ... yield chunks
}
```

### Configuration Files (YAML)

**YAML conventions:**
- Use comments extensively (explain each parameter)
- Include valid ranges and recommendations
- Group related settings
- Provide hardware-specific examples

**Pattern:**
```yaml
# Section description
section_name:
  # Parameter description (range: X-Y, default: Z)
  parameter_name: value

  # Hardware-specific recommendations:
  # - RTX 3090/4090 (24GB): value1
  # - A100 40GB: value2
  # - A100 80GB: value3
  another_parameter: value
```

---

## Key Files and Locations

### Critical Files (Do Not Break!)

These files are foundational - changes require extreme care:

1. **`src/havoc_core/config.py`**
   - All configuration dataclasses
   - Used throughout entire codebase
   - Breaking changes cascade everywhere

2. **`src/havoc_core/model/transformer.py`**
   - Main model class
   - Training and inference depend on this
   - Changes affect checkpoints

3. **`src/havoc_training/trainer.py`**
   - Training orchestration
   - Checkpoint format defined here
   - Changes can break resume functionality

4. **`src/havoc_inference/server.py`**
   - API contract
   - Frontend depends on endpoints
   - Breaking changes require frontend updates

5. **`frontend/src/lib/api.js`**
   - API client contract
   - Must match server endpoints
   - Breaking changes affect all components

### Configuration Hierarchy

Configs are loaded in this order (later overrides earlier):

1. **Default config in code** (`src/havoc_core/config.py`)
2. **YAML config file** (`configs/**/*.yaml`)
3. **Command-line arguments** (e.g., `--batch-size 4`)

**Example:**
```bash
# Uses all defaults
python scripts/train.py

# Uses YAML config
python scripts/train.py --config configs/training/custom.yaml

# Overrides specific parameters
python scripts/train.py --config my_config.yaml --batch-size 2 --learning-rate 1e-4
```

### Where Things Live

**Model weights & checkpoints:**
- Saved to: `checkpoints/checkpoint_step_N/`
- Contains: `model.pt`, `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `training_state.json`, `config.json`

**Logs:**
- Training: `logs/train.log`
- Inference: `logs/inference.log` (if configured)

**Data:**
- Expected: `data/math/`, `data/stats/`, `data/engineering/`, `data/general/`
- Format: Plain text files (`.txt`)

**Frontend build:**
- Development: `npm run dev` → `http://localhost:3000`
- Production: `npm run build` → `frontend/dist/`

---

## Common Tasks

### Task 1: Train from Scratch

```bash
# 1. Prepare data
mkdir -p data/math data/stats data/engineering data/general
# Add .txt files to each directory

# 2. (Optional) Train tokenizer
python -m havoc_core.tokenizer.train_tokenizer

# 3. Train model
python scripts/train.py --config configs/training/default_training.yaml

# 4. Monitor
tail -f logs/train.log
```

**Key files:**
- `scripts/train.py` - Entry point
- `configs/training/default_training.yaml` - Configuration
- `src/havoc_training/trainer.py` - Training logic

### Task 2: Resume Training

```bash
# Resume from checkpoint
python scripts/train.py --resume checkpoints/checkpoint_step_5000

# Resume and override parameters
python scripts/train.py --resume checkpoints/checkpoint_step_5000 --learning-rate 1e-5
```

**Note:** Resumes optimizer state, scheduler, scaler, and training state.

### Task 3: Run Inference Server

```bash
# Start server
python scripts/serve.py --checkpoint checkpoints/checkpoint_step_10000

# Custom config
python scripts/serve.py --checkpoint path/to/checkpoint --config configs/inference/custom.yaml --port 8001

# Access docs
open http://localhost:8000/docs
```

**Key files:**
- `scripts/serve.py` - Entry point
- `src/havoc_inference/server.py` - FastAPI app
- `src/havoc_inference/engine.py` - Generation logic

### Task 4: Run Chat UI

```bash
# Install dependencies (first time only)
cd frontend
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

**Access:** `http://localhost:3000`

**Key files:**
- `frontend/src/App.jsx` - Main application
- `frontend/src/lib/api.js` - API client
- `frontend/src/components/*.jsx` - UI components

### Task 5: Add a New Model Configuration

```bash
# 1. Copy existing config
cp configs/model/havoc_7b.yaml configs/model/havoc_3b.yaml

# 2. Edit parameters
# Modify num_layers, d_model, num_heads, etc.

# 3. Use in training
python scripts/train.py --config configs/training/my_training.yaml
# (Update my_training.yaml to reference havoc_3b.yaml)
```

### Task 6: Modify Generation Behavior

**For API-level changes:**

Edit `src/havoc_inference/engine.py:InferenceEngine.generate()` or `.generate_stream()`

**For sampling changes:**

Edit `src/havoc_inference/engine.py:InferenceEngine._sample_token()`

**For UI-level changes:**

Edit `frontend/src/components/Settings.jsx` (sliders, presets)

### Task 7: Add a New Reasoning Stage to SRS

```bash
# 1. Create new stage module
touch src/havoc_srs/my_new_stage.py

# 2. Implement stage interface
# See existing stages (mode.py, ground.py, etc.) for pattern

# 3. Update orchestrator
# Edit src/havoc_srs/orchestrator.py to include new stage

# 4. Update config
# Edit configs/srs/default_srs.yaml

# 5. Test
python scripts/demo_run.py
```

### Task 8: Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

---

## Testing and Validation

### Test Structure

**Unit tests:** `tests/`
- `test_config.py` - Config loading and validation
- `test_srs_pipeline.py` - SRS orchestrator
- `conftest.py` - PyTest fixtures

**Test pattern:**
```python
import pytest
from havoc_core.config import HavocConfig

def test_config_loading():
    """Test that config loads with defaults."""
    config = HavocConfig()
    assert config.d_model == 4096
    assert config.num_layers == 32
```

### Smoke Tests

**Quick validation that everything works:**

```bash
# 1. Config loading
python -c "from havoc_core.config import HavocConfig; print(HavocConfig())"

# 2. Model initialization
python -c "from havoc_core.model import HavocModel; from havoc_core.config import HavocConfig; m = HavocModel(HavocConfig()); print(m)"

# 3. Training (10 steps)
python scripts/train.py --max-steps 10

# 4. Inference server
python scripts/serve.py --checkpoint checkpoints/checkpoint_step_10 &
curl http://localhost:8000/health
```

### Validation Metrics

**Training:**
- Loss should decrease over time
- Perplexity should decrease (target: 10-50)
- Gradients should not explode (clip at 1.0)

**Inference:**
- Generation should complete without errors
- Streaming should produce tokens progressively
- Temperature 0.0 should be deterministic

**UI:**
- Should connect to API within 5 seconds
- Streaming should render token-by-token
- 3D background should render without lag

---

## Deployment Considerations

### Hardware Requirements

**Training (7B model):**
- Minimum: RTX 3090 (24GB) with batch_size=2, gradient_accumulation=16
- Recommended: A100 40GB with batch_size=4, gradient_accumulation=8
- Optimal: A100 80GB with batch_size=8, gradient_accumulation=4

**Inference:**
- Minimum: RTX 3080 (10GB) - single user
- Recommended: RTX 4090 (24GB) or A100 40GB - multiple concurrent users
- Optimal: A100 80GB - high throughput serving

**Frontend:**
- Any modern machine (CPU only)
- Node.js 18+ required

### Scaling Considerations

**Current limitations (single-GPU only):**
- No distributed training (DDP/FSDP)
- No model parallelism
- No tensor parallelism

**To add multi-GPU support:**
1. Wrap model in `DistributedDataParallel`
2. Use `torch.distributed` for initialization
3. Update `Trainer` to handle distributed training
4. Update configs to specify world_size, rank, etc.

**To add inference scaling:**
1. Deploy multiple inference servers
2. Add load balancer (nginx, HAProxy)
3. Add API gateway for auth/rate limiting
4. Consider vLLM or TensorRT-LLM for optimization

### Environment Variables

**Backend:**
```bash
CUDA_VISIBLE_DEVICES=0  # GPU selection
OMP_NUM_THREADS=8       # CPU threads
```

**Frontend:**
```bash
VITE_API_URL=http://localhost:8000  # API endpoint
```

### Docker Deployment (Future)

**Recommended Dockerfile structure:**
```dockerfile
# Backend
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN pip install -e .
CMD ["python", "scripts/serve.py", "--checkpoint", "checkpoints/..."]

# Frontend
FROM node:18-alpine
WORKDIR /app
COPY frontend/ .
RUN npm install && npm run build
CMD ["npm", "run", "preview"]
```

---

## Important Context

### Design Philosophy

**From the training guide:**
> "This training stack is production-ready but not magic. Real considerations: Training a 7B model from scratch is expensive (~100B tokens, ~1000 A100 GPU-hours). You'll need real data. Hyperparameters matter. Hardware constraints are real."

**No sycophancy:** The documentation is honest about limitations and requirements.

### Unique Features

What makes this codebase special:

1. **Complete system:** Not just model code - includes training, serving, UI, docs
2. **From scratch:** No dependency on Hugging Face Transformers
3. **Domain-specialized:** Math/stats/engineering focus
4. **SRS reasoning:** 8-stage explicit reasoning pipeline
5. **Production-ready:** Error handling, logging, checkpointing, resume
6. **Well-documented:** 2000+ lines of documentation

### Known Limitations

**What's missing:**
- ❌ Trained weights (you must train)
- ❌ Multi-GPU support (single GPU only)
- ❌ Experiment tracking (no W&B/TensorBoard)
- ❌ Docker deployment configs
- ❌ Authentication/authorization
- ❌ Real tokenizer model (uses dummy currently)

**What's scaffolded but incomplete:**
- ⚠️ RAG layer (needs document corpus)
- ⚠️ Evaluation harness (needs benchmark datasets)
- ⚠️ DSL executor (partial implementation)

### Troubleshooting Tips

**Out of Memory (OOM):**
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps` (keeps effective batch size)
- Enable mixed precision (`use_amp: true`)
- Reduce `max_seq_len` if needed

**Slow training:**
- Enable mixed precision
- Use bfloat16 on A100/H100
- Increase batch_size if memory allows
- Consider `torch.compile()` (PyTorch 2.0+)

**NaN loss:**
- Reduce learning rate
- Enable gradient clipping
- Use bfloat16 instead of float16
- Check data for corrupted samples

**API connection issues:**
- Verify server is running (`curl http://localhost:8000/health`)
- Check CORS settings in `server.py`
- Check firewall/port forwarding
- Update `VITE_API_URL` in frontend

**Frontend 3D issues:**
- Check WebGL support (`chrome://gpu`)
- Update graphics drivers
- Disable 3D background if needed (comment out `<Scene3D />` in `App.jsx`)

---

## Quick Reference

### Command Cheatsheet

```bash
# Training
python scripts/train.py
python scripts/train.py --config configs/training/my_config.yaml
python scripts/train.py --resume checkpoints/checkpoint_step_5000
python scripts/train.py --batch-size 2 --max-steps 10000

# Inference
python scripts/serve.py --checkpoint checkpoints/checkpoint_step_10000
python scripts/serve.py --checkpoint path/to/checkpoint --port 8001

# Frontend
cd frontend
npm install
npm run dev
npm run build

# Testing
pytest tests/
pytest -v tests/test_config.py

# CLI
python -m havoc_cli.main "Design a Box-Behnken DOE"

# Tokenizer training
python -m havoc_core.tokenizer.train_tokenizer
```

### File Quick Links

| Component | Main File | Config File |
|-----------|-----------|-------------|
| Model | `src/havoc_core/model/transformer.py` | `configs/model/havoc_7b.yaml` |
| Training | `src/havoc_training/trainer.py` | `configs/training/default_training.yaml` |
| Inference | `src/havoc_inference/server.py` | `configs/inference/default_inference.yaml` |
| Frontend | `frontend/src/App.jsx` | `frontend/vite.config.js` |
| SRS | `src/havoc_srs/orchestrator.py` | `configs/srs/default_srs.yaml` |
| Tools | `src/havoc_tools/python_math/engine.py` | `configs/tools/default_tools.yaml` |
| RAG | `src/havoc_rag/retrieval.py` | `configs/rag/default_rag.yaml` |

### Dependencies Quick Reference

**Python (core):**
- torch >= 2.1
- numpy >= 1.26
- scipy >= 1.11
- sympy >= 1.12
- statsmodels >= 0.14
- sentencepiece >= 0.1.99
- pyyaml >= 6.0

**Python (optional):**
- fastapi >= 0.104
- uvicorn[standard] >= 0.24
- pydantic >= 2.0
- faiss-cpu >= 1.7.4

**JavaScript:**
- react 18.2
- three 0.160
- @react-three/fiber 8.15
- react-markdown 9.0
- tailwindcss 3.4
- vite 5.0

### Configuration Parameter Reference

**Model (HavocConfig):**
- `num_layers`: 32 (number of transformer layers)
- `d_model`: 4096 (model dimension)
- `num_heads`: 32 (attention heads)
- `num_kv_heads`: 8 (KV heads for GQA)
- `head_dim`: 128 (dimension per head)
- `mlp_hidden_dim`: 11008 (MLP hidden size)
- `vocab_size`: 70000 (vocabulary size)
- `max_seq_len`: 4096 (max sequence length)

**Training (TrainingConfig):**
- `batch_size`: 2-8 (per-device batch size)
- `gradient_accumulation_steps`: 4-16 (effective batch multiplier)
- `learning_rate`: 3e-4 (peak LR)
- `min_lr`: 3e-5 (minimum LR for cosine schedule)
- `weight_decay`: 0.1 (AdamW weight decay)
- `max_grad_norm`: 1.0 (gradient clipping)
- `warmup_steps`: 2000 (LR warmup)
- `use_amp`: true (mixed precision)
- `amp_dtype`: "bfloat16" (or "float16")

**Inference (InferenceConfig):**
- `temperature`: 0.7 (sampling temperature)
- `top_p`: 0.9 (nucleus sampling)
- `top_k`: 50 (top-k sampling)
- `max_new_tokens`: 512 (max generation length)
- `repetition_penalty`: 1.1 (repetition penalty)

---

## Git Workflow

### Current Branch

This codebase is on branch: `claude/claude-md-midoin5rszc7fomc-01YYwnLcgKFF16emXiQ4drVF`

### Commit Strategy

**Good commit messages:**
- "Add mixed precision training support"
- "Fix KV-cache bug in generation"
- "Update frontend API client for new endpoints"

**Bad commit messages:**
- "Fix stuff"
- "WIP"
- "Update"

**When to commit:**
- After completing a logical unit of work
- After adding a new feature
- After fixing a bug
- Before attempting risky changes (safety commit)

### Branch Strategy

**Main branch:** Not specified (check `git remote show origin`)

**Feature branches:** Use descriptive names like:
- `feature/multi-gpu-training`
- `fix/generation-bug`
- `docs/update-training-guide`

---

## Additional Resources

**Documentation files:**
- `README.md` - Project overview
- `TRAINING_GUIDE.md` - Training workflow and tips
- `INFERENCE_GUIDE.md` - API reference and examples
- `COMPREHENSIVE_REPO_ANALYSIS.md` - Full codebase analysis
- `frontend/README.md` - Frontend documentation

**External references:**
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) (for LLM concepts)

---

## Contact and Support

**Issues:** This is a standalone repository. For issues, check existing documentation first.

**Updates:** Check git history for recent changes:
```bash
git log --oneline -10
```

---

**Last updated:** November 24, 2025
**For AI assistants:** This guide is comprehensive. Read relevant sections before making changes. When in doubt, check existing code patterns and follow established conventions.
