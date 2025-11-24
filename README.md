# 📘 HAVOC-7B / SIGMA-7B
### **Domain-Specialized 7B Transformer + SRS Reasoning Stack**

**HAVOC-7B (a.k.a. SIGMA-7B)** is a from-scratch 7B-parameter decoder-only transformer engineered for **math, statistics, engineering, DOE/SPC, and small-to-mid manufacturing intelligence**.

This repository provides:

- The model architecture
- Tokenizer pipeline
- Data ingestion + preprocessing
- RAG layer
- Math/Stats toolchain
- Full inference server
- Training stack
- Evaluation harness
- CLI
- The **SRS-7B reasoning pipeline**

**No weights included.**
Training loops and infrastructure exist — *you* train your model.

---

🧩 Repository Structure
src/
  havoc_core/        # model architecture, configs, tokenizer
  havoc_data/        # datasets & preprocessing
  havoc_tools/       # math/stats engine + DOE/SPC DSL
  havoc_rag/         # retrieval-augmented generation
  havoc_srs/         # Scott Reasoning Stack (8 stages)
  havoc_eval/        # benchmarks & evaluation harness
  havoc_cli/         # CLI interface
  havoc_training/    # full training pipeline
  havoc_inference/   # FastAPI inference server
frontend/            # React chat UI w/ 3D background
configs/             # YAML configs for model, training, inference, tools, RAG, SRS
scripts/             # train.py, serve.py, demo, dev checklist
tests/               # unit tests


---

## ⚙️ Core Model: HAVOC-7B

**Decoder-only transformer**

- 32 layers
- **d_model = 4096**
- **32 attention heads** (8 KV heads for GQA)
- **SwiGLU MLP (~11k hidden)**
- RoPE
- RMSNorm
- KV-cache optimized
- Greedy + sampling generation

Defined under:

`src/havoc_core/model/`

---

## 🔤 Tokenizer Pipeline

- SentencePiece trainer
- ~70–80k vocab
- Reserved tokens for DSL + SRS markers
- Domain symbol normalization

Train with:

`python -m havoc_core.tokenizer.train_tokenizer`

---

## 📚 Data Pipeline

- Weighted mixture of domain/general/dialog sources
- Preprocessing for math/engineering symbols
- PyTorch-style causal LM dataset

Configurable via:
`havoc_data/`
`configs/data/`

---

## 🔧 Math/Stats Tools + DSL

- T-tests, ANOVA, regression, DOE, derivatives
- DOE/SPC DSL → parsed → executed → typed results

Located in:

`havoc_tools/`

---

## 🔍 RAG Layer

Lightweight retrieval-augmented generation:

- Embeddings wrapper
- In-memory vector index
- Top-k search
- Grounding injection

Directory:

`havoc_rag/`


---

## 🧠 SRS-7B (Scott Reasoning Stack)

Eight explicit reasoning stages:

**MODE → GROUND → PLAN → EXECUTE → ARGUE → ARBITER → AUDIT → ANSWER**

Orchestrated through:



`havoc_srs/orchestrator.py`


Designed for **traceable reasoning**, **tool usage**, and **engineering/math accuracy**.

---

## 🧪 Evaluation Harness

- Benchmark registry
- Smoke tests
- Domain evals

Location:



`havoc_eval/`


---

## 🖥️ Training Pipeline

Full stack:

- AdamW (decoupled)
- Cosine/linear LR schedules w/ warmup
- Mixed precision (AMP, bfloat16 preferred)
- Gradient accumulation
- Checkpoint save/load
- Validation loop w/ perplexity

Run training:

`python scripts/train.py --config configs/training/default_training.yaml`


---

## 🌐 Inference Server (FastAPI)

Features:

- Text completion + chat
- Streaming (SSE)
- Greedy, top-k, top-p, temperature
- Repetition penalty
- KV-caching
- Mixed precision

Start server:



`python scripts/serve.py --checkpoint checkpoints/your_step`


Swagger docs:



http://localhost:8000/docs


---

## 💬 Frontend (React + Three.js)

- Modern chat UI
- Live streaming
- 3D animated background
- Token-by-token rendering
- Settings panel (temp / top-p / top-k sliders)

Start:



`cd frontend`
`npm install`
`npm run dev`


---

## 🚀 Quick Start

### Install


`pip install -e .`


### Train


`python scripts/train.py`


### Serve


`python scripts/serve.py --checkpoint path/to/checkpoint`


### Run CLI


`python -m havoc_cli.main` "Run a two-sample t-test between A and B"


---

## 🗺️ What’s Included vs Missing

### ✔️ Included
- Model implementation
- Training loop
- Inference server
- UI
- RAG
- Math/stats tools
- SRS-7B reasoning
- Full configs & documentation

### ❌ Not Included
- Trained weights
- Tokenizer model (you train it)
- Real corpora
- Multi-GPU training (single GPU only)

---

## 📄 License

**Apache-2.0**

---
