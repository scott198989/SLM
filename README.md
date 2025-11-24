# 🧠 HAVOC-7B / SIGMA-7B  
### A From-Scratch 7B Specialist Model for Math, Statistics, Six Sigma, DOE, SPC, and Process Engineering

Welcome to **HAVOC-7B** (and the upcoming **SIGMA-7B**).  
This repository contains all scaffolding, architecture, tooling, and reasoning infrastructure required to build a **from-scratch, domain-specialist 7B transformer** — **but NOT the training code or weights yet**.

HAVOC-7B is designed to punch far above its size through:

- A custom 7B transformer architecture  
- A domain-aware tokenizer  
- External math/stats/DOE/SPC tools  
- Retrieval-augmented reasoning (RAG)  
- A multi-stage reasoning framework (SRS-7B)

Think of it as a **math/stats/engineering sniper rifle**, not a general chatbot.

---

## 🔥 Project Goals

HAVOC-7B aims to:

- Outperform larger general models in **math, statistics, engineering, DOE, SPC, and Six Sigma**  
- Serve as a **local, offline specialist assistant**  
- Showcase how a small model + the right scaffolding can rival much larger models  
- Train exclusively on:
  - textbooks  
  - engineering/physics notes  
  - open datasets  
  - statistical and DOE handbooks  
  - domain DSLs  
  - light general text  

This is a **true from-scratch model** — no LLaMA/Qwen checkpoints.

---

## 🏗️ Repository Structure

```bash
src/
havoc_core/
havoc_data/
havoc_tools/
havoc_rag/
havoc_srs/
havoc_cli/

docs/
AGENTS.md
REASONING.md
MODEL.md
TOKENIZER.md
DATA.md

configs/
model/
data/
tools/
srs/
rag/

tests/
scripts/
README.md
```


**src/** contains the code.  
**docs/** contains specifications.  
**configs/** holds structured configuration.  
**tests/** contains unit tests.  
**scripts/** contains dev tools.

---

## 🧬 1. HAVOC-7B Architecture

A modern 7B decoder-only transformer:

- **32 layers**  
- **d_model = 4096**  
- **32 attention heads**  
- **head_dim = 128**  
- **SwiGLU MLP (≈11008 dim)**  
- **Grouped-Query Attention (GQA)**  
- **RMSNorm**  
- **RoPE positional encoding**  
- **4k–8k context**

Architecture files:

- `src/havoc_core/model/transformer.py`  
- `src/havoc_core/model/blocks.py`  
- `src/havoc_core/config.py`

---

## ✂️ 2. Custom Tokenizer (Domain-Aware)

Tokenizer target size: **70k–80k**.

Training corpus includes:

- Statistics textbooks  
- DOE manuals (Box-Behnken, Taguchi, factorial designs)  
- Six Sigma / SPC handbooks  
- Engineering and physics notes  
- Symbolic math  
- Light general text  

Preserves domain terms as whole tokens, e.g.:

- `ANOVA`, `p-value`, `Cpk`, `Ppk`  
- `factorial_doE`, `control_chart`  
- Unicode math symbols: `µ`, `σ`, `Σ`, `∂`, etc.

Special DSL tokens such as:

- `RUN_TTEST`  
- `DESIGN_DOE`  
- `CHECK_SPC`  
- `ALPHA_0_05`

Tokenizer code lives in:

- `src/havoc_core/tokenizer/`

---

## ⚙️ 3. Tools Layer (Math/Stats/DOE/SPC Engine)

HAVOC-7B does not guess numeric answers.  
It writes code → tools execute → model interprets.

Tools include:

- NumPy  
- SciPy  
- SymPy  
- statsmodels  
- custom DOE/SPC routines

Examples:

```python
run_ttest()
run_anova()
fit_regression()
evaluate_doe()
```

### Located in:

`src/havoc_tools/python_math/`

`src/havoc_tools/dsl/`

### 🔎 4. RAG Layer

The model accesses external references:

- PDFs

- textbooks

- engineering notes

- statistical formulas

- DOE/SPC theory

### Components:

- Embedding model wrapper

- Vector index (FAISS or pluggable)

- Retrieval interface

### Located in:

`src/havoc_rag/`

### 🧠 5. SRS-7B — Scott Reasoning Stack

This is the reasoning engine behind HAVOC-7B.

- MODE → GROUND → PLAN → EXECUTE → ARGUE → ARBITER → AUDIT → ANSWER

### MODE

- Classifies domain, difficulty, risk.

### GROUND

- Retrieve evidence through RAG.

### PLAN

- Produce structured steps + tool definitions.

### EXECUTE

- Run math/statistics tools & DSL commands.

### ARGUE

- Generate PRO and CON chains for high-risk tasks.

### ARBITER

- Decide outcome based on tool results + arguments.

### AUDIT

- Self-attack: check assumptions, errors, logic.

### ANSWER

- Final structured output with confidence & caveats.

Code: `src/havoc_srs/`

### 📊 6. Benchmark System

Benchmark suites will test:

- Pure math

- Probability/statistics

- ANOVA/regression

- DOE design + interpretation

- SPC / capability analysis

- Materials/process engineering

Three levels:

- Base model

- Model + tools

- Full SRS-7B pipeline

Located in:

`src/havoc_eval/`

### 🧩 7. Project Status
✔️ Completed / Scaffolded

- Repo layout

- Architecture definitions

- Tokenizer design

- Tools/DSL structure

- RAG interfaces

- SRS pipeline design

### 🟡 Work in Progress

- Implementing modules

Adding tests

- Writing detailed docs

### ⛔ Not Started Yet

- Model initialization

- Tokenizer training

- Pretraining

- Fine-tuning

- Checkpoints

- Distributed training setup

### 🏁 8. How to Use This Repo

For now:

- Explore the scaffolding

- Test the orchestrator with dummy data

- Customize configs

- Fill in tools and reasoning components

- Once scaffolding is complete:

- Train tokenizer

- Build dataset

- Begin pretraining (separate repo or directory)

- Insert trained weights into HAVOC-7B

- Enable full SRS-7B reasoning

### 🚀 9. Why HAVOC-7B Will Outperform Larger Models

- Domain-aware tokenizer

- External numeric tools

- Retrieval-backed reasoning

- Explicit multi-stage logic

- Built-in argument attack/defense

- Structured uncertainty

- Engineering DSLs

- Training on specialist corpora

HAVOC-7B doesn’t aim to be universal —
it aims to dominate its chosen domains.

### 🤝 10. Contributing

Codex 5.1 is the automated engineer for this project.
Human PRs should follow:

- modular code

- strong typing

- tests

- clear comments

- no training-related code here

- follow the docs in `/docs/`
