#!/usr/bin/env python3
"""
HAVOC Sanity Check Fine-tune Script

Purpose: Verify if checkpoint weights can stabilize with minimal fine-tuning.
If the model produces coherent completions after 100-300 steps → weights are stable.
If still nonsense → restart Phase 1 with corrected LR schedule.

Usage:
    python scripts/sanity_finetune.py \
        --checkpoint /workspace/SLM/checkpoints/phase1_h200_safe/checkpoint_step_750 \
        --data /workspace/SLM/data/sanity_check_50.jsonl \
        --steps 200 \
        --eval-every 50
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedTokenizerFast

# -------------------------
# CONFIGURATION
# -------------------------
DEFAULT_CHECKPOINT = "/workspace/SLM/checkpoints/phase1_h200_safe/checkpoint_step_750"
DEFAULT_TOKENIZER = "/workspace/SLM/artifacts/tokenizer"
DEFAULT_DATA = "/workspace/SLM/data/sanity_check_50.jsonl"

SANITY_CONFIG = {
    "max_steps": 200,
    "eval_every": 50,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "learning_rate": 5e-5,  # Very low LR for sanity check
    "max_seq_len": 256,
    "max_new_tokens": 64,
}

# -------------------------
# DATASET
# -------------------------
class SanityDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get("text", "")
                if text:
                    self.samples.append(text)

        print(f"Loaded {len(self.samples)} sanity check samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return input_ids, attention_mask


# -------------------------
# EVALUATION PROMPTS
# -------------------------
EVAL_PROMPTS = [
    "Question: What is 2 + 2?\nAnswer:",
    "Question: What is the capital of France?\nAnswer:",
    "Question: What is the chemical symbol for water?\nAnswer:",
    "Question: What does CPU stand for?\nAnswer:",
    "Question: What is 7 × 8?\nAnswer:",
]


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64, device: str = "cuda"):
    """Generate text from a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # Simple greedy decoding
        generated = inputs.clone()
        for _ in range(max_new_tokens):
            outputs, _ = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def evaluate_coherence(model, tokenizer, device: str = "cuda"):
    """Run evaluation prompts and check for coherent responses."""
    print("\n" + "=" * 60)
    print("COHERENCE EVALUATION")
    print("=" * 60)

    results = []
    for prompt in EVAL_PROMPTS:
        response = generate_text(model, tokenizer, prompt, max_new_tokens=32, device=device)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        results.append({"prompt": prompt, "response": response})

    print("=" * 60 + "\n")
    return results


def sanity_finetune(
    checkpoint_path: str,
    tokenizer_path: str,
    data_path: str,
    max_steps: int = 200,
    eval_every: int = 50,
):
    """Run sanity fine-tune to test weight stability."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Load tokenizer
    # -------------------------
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # -------------------------
    # Load model
    # -------------------------
    checkpoint_dir = Path(checkpoint_path)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"

    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Dynamically import the right config/model classes
    # Try HavocPrimeModel first (7B), fall back to HavocModel (2B)
    try:
        from havoc_core.config_7b import Havoc7BConfig
        from havoc_core.model.prime_model import HavocPrimeModel
        config = Havoc7BConfig.from_json_file(str(config_path))
        model = HavocPrimeModel(config)
        print("Using HavocPrimeModel (7B architecture)")
    except Exception as e:
        print(f"Could not load 7B model ({e}), trying 2B...")
        from havoc_core.config import HavocConfig
        from havoc_core.model.transformer import HavocModel
        model_cfg = config_dict.get("model_config", config_dict)
        config = HavocConfig(**{k: v for k, v in model_cfg.items() if k in HavocConfig.__dataclass_fields__})
        model = HavocModel(config)
        print("Using HavocModel (2B architecture)")

    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model parameters: {param_count:.2f}B")

    # -------------------------
    # Check embedding compatibility
    # -------------------------
    model_vocab = config_dict.get("model_config", config_dict).get("vocab_size", 70000)
    tokenizer_vocab = tokenizer.vocab_size
    print(f"\nVocab check: model={model_vocab}, tokenizer={tokenizer_vocab}")

    if model_vocab != tokenizer_vocab:
        print(f"WARNING: Vocab mismatch! Model expects {model_vocab}, tokenizer has {tokenizer_vocab}")
        print("This may cause index errors or degraded performance.")

    # -------------------------
    # Initial evaluation (before fine-tune)
    # -------------------------
    print("\n>>> INITIAL STATE (before sanity fine-tune)")
    evaluate_coherence(model, tokenizer, device)

    # -------------------------
    # Setup training
    # -------------------------
    dataset = SanityDataset(data_path, tokenizer, max_len=SANITY_CONFIG["max_seq_len"])
    loader = DataLoader(dataset, batch_size=SANITY_CONFIG["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SANITY_CONFIG["learning_rate"],
        weight_decay=0.01
    )
    scaler = GradScaler()

    # -------------------------
    # Training loop
    # -------------------------
    print(f"\nStarting sanity fine-tune for {max_steps} steps...")
    print(f"Learning rate: {SANITY_CONFIG['learning_rate']}")
    print(f"Batch size: {SANITY_CONFIG['batch_size']} × {SANITY_CONFIG['gradient_accumulation']} = {SANITY_CONFIG['batch_size'] * SANITY_CONFIG['gradient_accumulation']}")

    model.train()
    global_step = 0
    accumulated_loss = 0.0
    accum_count = 0

    while global_step < max_steps:
        for batch in loader:
            if global_step >= max_steps:
                break

            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Shift for causal LM
            labels = input_ids[:, 1:].contiguous()
            input_ids_shifted = input_ids[:, :-1].contiguous()
            attention_mask_shifted = attention_mask[:, :-1].contiguous()

            with autocast(dtype=torch.float16):
                logits, _ = model(input_ids_shifted, attention_mask=attention_mask_shifted)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=0  # pad_token_id
                )
                loss = loss / SANITY_CONFIG["gradient_accumulation"]

            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * SANITY_CONFIG["gradient_accumulation"]
            accum_count += 1

            if accum_count % SANITY_CONFIG["gradient_accumulation"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accumulated_loss / SANITY_CONFIG["gradient_accumulation"]
                accumulated_loss = 0.0

                if global_step % 10 == 0:
                    print(f"Step {global_step:4d} | Loss: {avg_loss:.4f}")

                # Evaluate periodically
                if global_step % eval_every == 0:
                    print(f"\n>>> EVALUATION at step {global_step}")
                    evaluate_coherence(model, tokenizer, device)
                    model.train()

    # -------------------------
    # Final evaluation
    # -------------------------
    print("\n>>> FINAL STATE (after sanity fine-tune)")
    results = evaluate_coherence(model, tokenizer, device)

    # -------------------------
    # Verdict
    # -------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECK VERDICT")
    print("=" * 60)
    print("""
Review the outputs above:

✓ PASS (weights stable) if:
  - Responses are semantically related to questions
  - Arithmetic questions get numeric answers
  - Factual questions get factual responses
  - No severe repetition loops or garbage

✗ FAIL (weights unstable) if:
  - Responses are random/unrelated tokens
  - Severe repetition (same token repeated)
  - Complete nonsense or semantic drift
  - Loss did not decrease during training

If PASS → Proceed to Phase 2 with current checkpoint
If FAIL → Restart Phase 1 with corrected hyperparameters
""")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAVOC Sanity Fine-tune Test")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint directory")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help="Path to tokenizer")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Path to sanity check JSONL")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of fine-tune steps")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N steps")

    args = parser.parse_args()

    sanity_finetune(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        data_path=args.data,
        max_steps=args.steps,
        eval_every=args.eval_every,
    )
