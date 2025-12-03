# HAVOC-7B + PRIME: Complete Architecture Specification

**Version:** 2.0
**Date:** December 2, 2025
**Authors:** Architecture Team
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [7B Model Configuration](#7b-model-configuration)
3. [Architecture Components](#architecture-components)
4. [Reasoning Token System](#reasoning-token-system)
5. [PRIME Meta-Reasoning Integration](#prime-meta-reasoning-integration)
6. [Tool-Calling System](#tool-calling-system)
7. [Training Infrastructure](#training-infrastructure)
8. [Memory Optimization](#memory-optimization)
9. [Code Implementation](#code-implementation)
10. [Usage Examples](#usage-examples)

---

## Overview

### System Architecture

HAVOC-7B is a **7 billion parameter decoder-only transformer** with integrated PRIME meta-reasoning:

```
┌─────────────────────────────────────────────────────────────┐
│                      HAVOC-7B PRIME                          │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Base Transformer (7B Parameters)                     │  │
│  │  • 32 Layers                                          │  │
│  │  • d_model: 4096                                      │  │
│  │  • 32 Attention Heads (8 KV Heads - GQA)             │  │
│  │  • SwiGLU MLP (11008 hidden dim)                     │  │
│  │  • RoPE Positional Embeddings                        │  │
│  │  • RMSNorm                                           │  │
│  │  • FlashAttention-2 Support                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↕                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  PRIME Meta-Reasoning Layer                          │  │
│  │  • Budget Router (MICRO/LIGHT/MEDIUM/HEAVY)         │  │
│  │  • Operator Graph Builder                           │  │
│  │  • Adversarial Reasoning (Advocate/Attack/Pragmatist)│  │
│  │  • Chrono-Loop Refinement                           │  │
│  │  • Global Workspace Memory                          │  │
│  │  • Verification Layer                               │  │
│  │  • Compression Layer                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↕                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  SRS Tool Integration                                 │  │
│  │  • Python Math Engine (scipy/statsmodels)            │  │
│  │  • DSL Executor (DOE/SPC)                           │  │
│  │  • RAG Helper (FAISS retrieval)                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↕                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Reasoning Token System                               │  │
│  │  • <reason>...</reason> for chain-of-thought         │  │
│  │  • <tool>...</tool> for tool calls                   │  │
│  │  • <advocate>, <attack>, <pragmatist> for PRIME     │  │
│  │  • Visible, not hidden or compressed                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Visible Reasoning**: All reasoning is explicit in the token stream
2. **Budget-Based Computation**: Adaptive reasoning depth based on task complexity
3. **Tool Integration**: Native tool-calling support at model level
4. **Local Training**: Optimized for single-GPU training (RTX 5090 24GB)
5. **Modular Architecture**: Components can be modified independently

---

## 7B Model Configuration

### Parameter Count Breakdown

```python
# Embedding Layer
vocab_size × d_model = 70,000 × 4,096 = 286,720,000

# Transformer Layers (32 layers)
Per-layer parameters:
  - Attention:
    • Q proj: 4096 × (32 × 128) = 16,777,216
    • K proj: 4096 × (8 × 128)  = 4,194,304
    • V proj: 4096 × (8 × 128)  = 4,194,304
    • O proj: (32 × 128) × 4096 = 16,777,216
    • Subtotal: 41,943,040

  - MLP (SwiGLU):
    • W1: 4096 × 11008 = 45,088,768
    • W2: 4096 × 11008 = 45,088,768
    • W3: 11008 × 4096 = 45,088,768
    • Subtotal: 135,266,304

  - RMSNorm (2 layers): 2 × 4096 = 8,192

  Per-layer total: 177,217,536

Total for 32 layers: 5,670,961,152

# Output Layer (weight-tied)
0 (shares embedding weights)

# Final RMSNorm
4,096

# Total Parameters: ~6.96B ≈ 7B
```

### Model Configuration (Python)

```python
@dataclass
class Havoc7BConfig(HavocConfig):
    """HAVOC-7B Production Configuration"""

    # Model architecture
    vocab_size: int = 70000
    d_model: int = 4096
    num_layers: int = 32
    max_seq_len: int = 4096

    # Attention
    attention: AttentionConfig = field(default_factory=lambda: AttentionConfig(
        num_heads=32,
        num_kv_heads=8,  # GQA ratio 4:1
        head_dim=128,
        dropout=0.0,
        rotary_dim=128,
        rope_theta=10000.0,  # Standard RoPE base
        bias=False
    ))

    # MLP
    mlp: MLPConfig = field(default_factory=lambda: MLPConfig(
        hidden_dim=11008,  # ~2.7x expansion for SwiGLU
        activation="swiglu",
        bias=False
    ))

    # Regularization
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Reasoning tokens (new)
    reason_start_token_id: int = 10
    reason_end_token_id: int = 11
    tool_start_token_id: int = 12
    tool_end_token_id: int = 13
    advocate_token_id: int = 14
    attack_token_id: int = 15
    pragmatist_token_id: int = 16
```

---

## Architecture Components

### 1. Base Transformer (Upgraded to 7B)

The base transformer is a standard decoder-only architecture with:

- **32 transformer blocks**
- **Grouped-Query Attention (GQA)** with 32 query heads and 8 KV heads
- **Rotary Position Embeddings (RoPE)** with θ=10000
- **SwiGLU MLP** with ~2.7x hidden dimension expansion
- **RMSNorm** for layer normalization
- **Weight tying** between input embeddings and output layer

### 2. PRIME Meta-Reasoning System

PRIME is **integrated at the generation level**, not as a post-hoc wrapper. The system includes:

#### Budget Router

```python
class Budget(Enum):
    MICRO = "MICRO"    # Direct answer, no reasoning (trivial tasks)
    LIGHT = "LIGHT"    # Minimal reasoning, 1-3 subgoals
    MEDIUM = "MEDIUM"  # Standard reasoning, adversarial enabled
    HEAVY = "HEAVY"    # Full reasoning, chrono-loop + verification
```

**Routing Logic:**
- **MICRO**: Trivial arithmetic, definitions → Direct answer
- **LIGHT**: Simple questions → 1-3 subgoals, no adversarial
- **MEDIUM**: Stats/math problems → Full PRIME, limited chrono
- **HEAVY**: DOE/SPC/complex → Full PRIME + chrono-loop

#### Operator Graph

Dynamically built subgoal dependency graph based on task complexity.

#### Adversarial Reasoning

Three perspectives synthesized into final answer:
- **Advocate**: Optimistic argument for current solution
- **Attack**: Critical analysis, find flaws
- **Pragmatist**: Balanced view considering constraints

#### Chrono-Loop

Iterative refinement with noise injection for robustness.

#### Global Workspace

Stores facts, partial results, constraints during reasoning.

#### Verification Layer

Post-reasoning validation of consistency and correctness.

#### Compression Layer

Reduces workspace to essential facts for final answer.

---

## Reasoning Token System

### Special Tokens

HAVOC-7B uses explicit reasoning tokens to make chain-of-thought visible:

```
<reason> ... </reason>   → Chain-of-thought reasoning
<tool> ... </tool>       → Tool call specification
<advocate> ... </advocate> → Advocate's argument
<attack> ... </attack>   → Attack's counterargument
<pragmatist> ... </pragmatist> → Pragmatist's synthesis
```

### Vocabulary Extension

```python
SPECIAL_TOKENS = [
    "<pad>",       # 0
    "<bos>",       # 1
    "<eos>",       # 2
    "<unk>",       # 3
    "<mask>",      # 4 (for potential future use)
    # ... (5-9 reserved)
    "<reason>",    # 10
    "</reason>",   # 11
    "<tool>",      # 12
    "</tool>",     # 13
    "<advocate>",  # 14
    "</advocate>", # 15 (implicit from next token)
    "<attack>",    # 16
    "</attack>",   # 17
    "<pragmatist>",# 18
    "</pragmatist>",# 19
]
```

### Chain-of-Thought Format

**Example output:**

```
User: Design a 3-factor Box-Behnken DOE

<reason>
Task: Design Box-Behnken DOE with 3 factors.
Subgoal 1: Identify factor levels (typically 3: -1, 0, 1)
Subgoal 2: Generate Box-Behnken design matrix
Subgoal 3: Calculate required runs
</reason>

<tool>
{"tool": "dsl_executor", "args": {"operation": "box_behnken", "factors": 3}}
</tool>

<advocate>
Box-Behnken design is efficient: requires only 15 runs for 3 factors vs 27 for full factorial.
</advocate>

<attack>
Box-Behnken does not include corner points - may miss extreme interactions.
</attack>

<pragmatist>
For screening, Box-Behnken is appropriate. If extreme interactions are critical, use Central Composite Design instead.
</pragmatist>

Final answer: Box-Behnken design with 3 factors requires 15 runs...
```

**Critical**: These tokens are **NOT stripped** during generation. They remain in the output for full transparency.

---

## PRIME Meta-Reasoning Integration

### Architecture: Model-Level Integration

PRIME is integrated **inside** the model's generation loop, not as an external orchestrator.

```python
class HavocPrimeModel(nn.Module):
    """HAVOC-7B with integrated PRIME meta-reasoning"""

    def __init__(self, config: Havoc7BConfig):
        super().__init__()

        # Base transformer
        self.base_model = HavocModel(config)

        # PRIME components
        self.router = TaskRouter()
        self.graph_builder = OperatorGraphBuilder()
        self.workspace = GlobalWorkspace()
        self.advocate = Advocate()
        self.attack = HavocAttack()
        self.pragmatist = Pragmatist()
        self.synthesizer = AdversarialSynthesizer()
        self.verification = GlobalVerification()
        self.compression = FinalCompression()

        # Tool interface
        self.tool_registry = ToolRegistry()

    def generate_with_prime(
        self,
        prompt_ids: torch.Tensor,
        tools: Dict[str, Any],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        enable_prime: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with PRIME meta-reasoning.

        Returns dict with:
        - generated_ids: token IDs
        - text: decoded text (includes reasoning tokens)
        - workspace: final workspace state
        - verification: verification report
        """
        # Step 1: Route task
        prompt_text = self.tokenizer.decode(prompt_ids[0])
        routing = self.router.route(prompt_text)

        if not enable_prime or routing.budget == Budget.MICRO:
            # Direct generation without PRIME
            return self._direct_generate(prompt_ids, max_new_tokens, temperature)

        # Step 2: Build operator graph
        operator_graph = self.graph_builder.build_graph(
            prompt_text, routing.task_type, routing.budget
        )

        # Step 3: Generate with reasoning tokens
        return self._generate_with_reasoning(
            prompt_ids, operator_graph, routing, tools, max_new_tokens, temperature
        )
```

### Generation Flow with PRIME

```
1. User Prompt → Tokenize
2. Route Task → Determine Budget
3. Build Operator Graph → Subgoals
4. For each subgoal:
   a. Generate <reason> ... </reason>
   b. If tool needed: Generate <tool> ... </tool>
   c. Execute tool → Get result
   d. If adversarial enabled:
      - Generate <advocate> ... </advocate>
      - Generate <attack> ... </attack>
      - Generate <pragmatist> ... </pragmatist>
   e. Update workspace
5. Verification
6. Compression
7. Final answer generation
```

---

## Tool-Calling System

### Tool Call Format (JSON)

Tools are called using JSON embedded in `<tool>...</tool>` tokens:

```json
{"tool": "<tool_name>", "args": {<arguments>}}
```

### Supported Tools

#### 1. Python Math Engine

```json
{
  "tool": "python_math",
  "args": {
    "operation": "t_test",
    "group1": [1.2, 1.5, 1.3, 1.6],
    "group2": [2.1, 2.3, 2.0, 2.4]
  }
}
```

#### 2. DSL Executor (DOE/SPC)

```json
{
  "tool": "dsl_executor",
  "args": {
    "operation": "box_behnken",
    "factors": [
      {"name": "Temperature", "low": 100, "high": 200},
      {"name": "Pressure", "low": 10, "high": 50},
      {"name": "Speed", "low": 50, "high": 150}
    ]
  }
}
```

#### 3. RAG Helper

```json
{
  "tool": "rag_helper",
  "args": {
    "query": "What is Box-Behnken design?",
    "top_k": 3
  }
}
```

### Tool Registry

```python
class ToolRegistry:
    """Registry for available tools"""

    def __init__(self):
        self.tools = {
            "python_math": PythonMathEngine(),
            "dsl_executor": DSLExecutor(),
            "rag_helper": RAGHelper()
        }

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool = self.tools[tool_name]
        try:
            result = tool.execute(**args)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

---

## Training Infrastructure

### Phase 0: Tokenizer Training

**Goal:** Build SentencePiece tokenizer with 70k vocabulary including reasoning tokens.

**Data:** 10GB+ diverse text corpus

**Script:** `scripts/phase0_train_tokenizer.py`

```python
def train_tokenizer():
    config = TokenizerTrainingConfig(
        vocab_size=70000,
        model_type="bpe",
        special_tokens=[
            "<pad>", "<bos>", "<eos>", "<unk>",
            "<reason>", "</reason>",
            "<tool>", "</tool>",
            "<advocate>", "</advocate>",
            "<attack>", "</attack>",
            "<pragmatist>", "</pragmatist>"
        ],
        input_files=["data/corpus/*.txt"],
        output_dir="artifacts/tokenizer",
        character_coverage=0.9995,
        max_sentence_length=4096
    )

    train_sentencepiece(config)
```

**Output:** `tokenizer.model`, `tokenizer.vocab`

### Phase 1: Pretraining

**Goal:** Train base language model on 100B tokens

**Data Mixture:**
- 60% domain-specific (math, stats, engineering, DOE/SPC)
- 30% general knowledge
- 10% code/technical

**Configuration:**

```yaml
# configs/training/phase1_pretrain_7b.yaml

model:
  config_name: "havoc_7b"

data:
  sources:
    - path: "data/math"
      weight: 0.25
    - path: "data/stats"
      weight: 0.20
    - path: "data/engineering"
      weight: 0.15
    - path: "data/general"
      weight: 0.30
    - path: "data/code"
      weight: 0.10

  max_seq_len: 2048  # Start with 2048, extend later
  pack_sequences: true

training:
  # Hardware: RTX 5090 (24GB)
  batch_size: 1
  gradient_accumulation_steps: 32
  effective_batch_size: 32  # 1 × 32

  # Optimizer
  learning_rate: 3.0e-4
  min_learning_rate: 3.0e-5
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8

  # Schedule
  lr_scheduler_type: "cosine"
  warmup_steps: 2000
  max_steps: 100000  # ~100B tokens @ batch_size=32, seq_len=2048

  # Mixed precision
  use_amp: true
  amp_dtype: "bfloat16"  # RTX 5090 supports bf16

  # Gradient
  max_grad_norm: 1.0

  # Checkpointing
  checkpoint_dir: "checkpoints/phase1"
  save_every_n_steps: 5000
  keep_last_n_checkpoints: 3

  # Validation
  eval_every_n_steps: 1000
  eval_samples: 200

  # Logging
  log_every_n_steps: 10
  log_dir: "logs/phase1"

  # Memory optimization
  gradient_checkpointing: true
  cpu_offload: false
  use_flash_attention: true
```

**Expected Training Time:** ~1000 GPU-hours on RTX 5090 (~42 days)

**Checkpoints:** Every 5000 steps

### Phase 2: Supervised Fine-Tuning (SFT)

**Goal:** Teach reasoning and tool use

**Data:** 100k curated examples of:
- Chain-of-thought reasoning
- Tool calls with `<tool>...</tool>` format
- PRIME-style adversarial reasoning
- Math/stats problem solving

**Format:**

```json
{
  "prompt": "Design a Box-Behnken DOE for 3 factors",
  "completion": "<reason>\nTask: Design Box-Behnken...\n</reason>\n<tool>\n{\"tool\": \"dsl_executor\", ...}\n</tool>\n..."
}
```

**Configuration:**

```yaml
# configs/training/phase2_sft_7b.yaml

model:
  checkpoint_path: "checkpoints/phase1/checkpoint_step_100000"

data:
  sources:
    - path: "data/sft/reasoning_examples.jsonl"
      weight: 0.40
    - path: "data/sft/tool_use_examples.jsonl"
      weight: 0.30
    - path: "data/sft/prime_examples.jsonl"
      weight: 0.30

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  effective_batch_size: 32

  learning_rate: 1.0e-5  # Lower LR for fine-tuning
  min_learning_rate: 1.0e-6
  warmup_steps: 500
  max_steps: 10000

  # Other settings same as Phase 1
```

**Expected Training Time:** ~100 GPU-hours (~4 days)

### Phase 3: Conversational Polish

**Goal:** Polish conversational quality and safety

**Data:** 50k conversational examples with helpful, harmless, honest responses

**Configuration:**

```yaml
# configs/training/phase3_polish_7b.yaml

model:
  checkpoint_path: "checkpoints/phase2/checkpoint_step_10000"

data:
  sources:
    - path: "data/polish/conversations.jsonl"
      weight: 1.0

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 5.0e-6  # Even lower LR
  max_steps: 5000
```

**Expected Training Time:** ~50 GPU-hours (~2 days)

---

## Memory Optimization

### RTX 5090 (24GB VRAM) Configuration

#### Model Memory Footprint

```
Model Parameters: 7B × 2 bytes (bf16) = 14 GB
Optimizer State (AdamW): 7B × 8 bytes = 56 GB (!)
Gradients: 7B × 2 bytes = 14 GB
Activations: ~2-4 GB (depends on batch size & seq length)

Total without optimizer: ~30-32 GB → DOESN'T FIT
Total with optimizer: ~86 GB → DEFINITELY DOESN'T FIT
```

#### Optimization Strategy

1. **Mixed Precision (bf16)**: ✅ Reduces model weights to 14GB
2. **Gradient Accumulation**: ✅ Effective batch size with micro-batches
3. **Gradient Checkpointing**: ✅ Recompute activations instead of storing
4. **Flash Attention**: ✅ Reduces attention memory from O(n²) to O(n)
5. **CPU Offloading**: ⚠️ Optional (slow but enables training)

#### Final Configuration for RTX 5090

```python
@dataclass
class OptimizedTrainingConfig:
    """Memory-optimized config for RTX 5090 (24GB)"""

    # Batch size
    batch_size: int = 1  # CRITICAL: Keep at 1
    gradient_accumulation_steps: int = 32  # Effective batch = 32

    # Sequence length
    max_seq_len: int = 2048  # Start here, can extend to 4096 later

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # RTX 5090 supports bf16 natively

    # Gradient checkpointing
    gradient_checkpointing: bool = True  # CRITICAL: Must enable
    checkpoint_every_n_layers: int = 4  # Checkpoint every 4 layers

    # Flash Attention
    use_flash_attention: bool = True  # CRITICAL: Must enable

    # CPU offloading (optional)
    cpu_offload_optimizer: bool = False  # Try false first
    cpu_offload_params: bool = False

    # Optimizer
    optimizer: str = "adamw_fused"  # Fused AdamW for efficiency
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    # Gradient clipping
    max_grad_norm: float = 1.0
```

#### Memory Breakdown (Optimized)

```
Model (bf16): 14 GB
Gradients (bf16): 14 GB
Optimizer (offloaded or ZeRO-1): 0 GB (on CPU) or ~4GB (on GPU with ZeRO-1)
Activations (with checkpointing): ~2 GB
Flash Attention: ~1 GB
Misc buffers: ~1 GB

Total: ~18-22 GB → FITS IN 24GB!
```

### Advanced: DeepSpeed ZeRO-1

For even better memory efficiency:

```python
# deepspeed_config.json
{
  "zero_optimization": {
    "stage": 1,  # Partition optimizer states
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "bf16": {
    "enabled": true
  },
  "gradient_accumulation_steps": 32,
  "gradient_clipping": 1.0
}
```

---

## Code Implementation

*(Full code implementations will be provided in separate files)*

### Key Files to Create/Modify

1. **`src/havoc_core/config.py`**: Add `Havoc7BConfig`
2. **`src/havoc_core/model/prime_model.py`**: New `HavocPrimeModel`
3. **`src/havoc_core/reasoning_tokens.py`**: Reasoning token utilities
4. **`src/havoc_core/tool_interface.py`**: Tool registry and execution
5. **`src/havoc_training/optimized_trainer.py`**: Memory-optimized trainer
6. **`scripts/phase0_train_tokenizer.py`**: Tokenizer training
7. **`scripts/phase1_pretrain.py`**: Pretraining script
8. **`scripts/phase2_sft.py`**: SFT script
9. **`scripts/phase3_polish.py`**: Conversational polish script
10. **`configs/model/havoc_7b.yaml`**: 7B model config

---

## Usage Examples

### Example 1: Direct Generation (No PRIME)

```python
from havoc_core.model.prime_model import HavocPrimeModel
from havoc_core.config import Havoc7BConfig

# Load model
config = Havoc7BConfig()
model = HavocPrimeModel.from_pretrained("checkpoints/phase3/final")

# Simple generation
result = model.generate_with_prime(
    prompt="What is 2 + 2?",
    enable_prime=False,  # Disable PRIME for trivial task
    max_new_tokens=50
)

print(result["text"])
# Output: "2 + 2 = 4"
```

### Example 2: PRIME Reasoning (MEDIUM Budget)

```python
# Stats question
result = model.generate_with_prime(
    prompt="Compare two groups using t-test: [1,2,3,4,5] vs [2,3,4,5,6]",
    enable_prime=True,
    tools={
        "python_math": PythonMathEngine()
    },
    max_new_tokens=512
)

print(result["text"])
# Output:
# <reason>
# Task: Compare two groups using t-test
# Subgoal 1: Extract data
# Subgoal 2: Run t-test
# Subgoal 3: Interpret result
# </reason>
#
# <tool>
# {"tool": "python_math", "args": {"operation": "t_test", ...}}
# </tool>
#
# Result: t-statistic = -2.236, p-value = 0.048
# Conclusion: Groups are significantly different at α=0.05
```

### Example 3: Full PRIME (HEAVY Budget)

```python
# Complex DOE design
result = model.generate_with_prime(
    prompt="Design a Box-Behnken DOE for temperature (100-200°C), pressure (10-50 bar), and speed (50-150 RPM)",
    enable_prime=True,
    tools={
        "dsl_executor": DSLExecutor(),
        "rag_helper": RAGHelper()
    },
    max_new_tokens=1024
)

print(result["text"])
# Output: (full PRIME reasoning with <advocate>, <attack>, <pragmatist>, etc.)

# Access structured results
print(result["workspace"].summarize())
print(result["verification"])
```

---

## Next Steps

1. **Review this architecture specification**
2. **Implement code files** (see Code Implementation section)
3. **Prepare training data** for Phase 0-3
4. **Train tokenizer** (Phase 0)
5. **Start pretraining** (Phase 1)

---

**End of Architecture Specification**
