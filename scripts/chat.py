#!/usr/bin/env python3
"""
HAVOC Interactive Chat Script

Loads the HAVOC model and tokenizer for interactive generation.
Uses the same tokenizer and configuration as training for consistency.

Usage:
    python scripts/chat.py --checkpoint /path/to/checkpoint

Environment:
    - Expects tokenizer at /workspace/SLM/artifacts/tokenizer
    - Automatically detects HavocPrimeModel (7B) vs HavocModel (2B)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# =============================================================================
# CONFIGURATION (RunPod: AMD MI300X ROCm)
# =============================================================================
# Repo root: /workspace/SLM
# Dataset root: /workspace/data
# Checkpoints: /workspace/SLM/checkpoints/havoc_7b_phase1
DEFAULT_CHECKPOINT = "/workspace/SLM/checkpoints/havoc_7b_phase1"
DEFAULT_TOKENIZER = "/workspace/SLM/artifacts/tokenizer"

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50

# =============================================================================
# TOKENIZER LOADING - Uses HavocTokenizer for consistency with training
# =============================================================================
def load_tokenizer(tokenizer_path: str):
    """Load tokenizer - tries HavocTokenizer first, falls back to HF."""
    tokenizer_path = Path(tokenizer_path)

    # Try HavocTokenizer (SentencePiece-based)
    spm_model = tokenizer_path / "tokenizer.model"
    if spm_model.exists():
        try:
            # Add src to path
            src_path = Path(__file__).parent.parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from havoc_core.tokenizer.tokenizer import HavocTokenizer
            print(f"Loading HavocTokenizer from: {tokenizer_path}")
            return HavocTokenizer(str(tokenizer_path))
        except Exception as e:
            print(f"Failed to load HavocTokenizer: {e}")

    # Fallback to HuggingFace PreTrainedTokenizerFast
    try:
        from transformers import PreTrainedTokenizerFast
        print(f"Loading PreTrainedTokenizerFast from: {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        # Ensure token IDs are set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 1
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 2
        return tokenizer
    except Exception as e:
        print(f"Failed to load HuggingFace tokenizer: {e}")
        raise RuntimeError(f"Could not load tokenizer from {tokenizer_path}")


# =============================================================================
# MODEL LOADING
# =============================================================================
def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # If it's already a specific checkpoint (has config.json), return it
    if (checkpoint_dir / "config.json").exists():
        return checkpoint_dir

    # Otherwise, find the latest checkpoint_step_N
    checkpoints = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint_step_"):
            try:
                step = int(d.name.split("_")[-1])
                checkpoints.append((step, d))
            except ValueError:
                continue

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Return the one with highest step number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest = checkpoints[0][1]
    print(f"Found latest checkpoint: {latest.name}")
    return latest


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint, auto-detecting architecture."""
    # Find latest checkpoint if directory given
    checkpoint_dir = find_latest_checkpoint(checkpoint_path)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Detect model architecture from config
    model_cfg = config_dict.get("model_config", config_dict)
    num_layers = model_cfg.get("num_layers", 20)
    d_model = model_cfg.get("d_model", 2560)

    # 7B config has ~32 layers, 4096 d_model
    # 2B config has ~20 layers, 2560 d_model
    is_7b = num_layers >= 28 or d_model >= 4096

    if is_7b:
        try:
            from havoc_core.config_7b import Havoc7BConfig
            from havoc_core.model.prime_model import HavocPrimeModel

            print(f"Detected 7B architecture (layers={num_layers}, d_model={d_model})")
            config = Havoc7BConfig.from_pretrained(str(checkpoint_dir))
            model = HavocPrimeModel(config)
        except ImportError:
            print("Warning: HavocPrimeModel not available, trying HavocModel")
            is_7b = False

    if not is_7b:
        from havoc_core.config import HavocConfig
        from havoc_core.model.transformer import HavocModel

        print(f"Using 2B architecture (layers={num_layers}, d_model={d_model})")
        # Build config from dict
        config = HavocConfig(
            vocab_size=model_cfg.get("vocab_size", 70000),
            d_model=model_cfg.get("d_model", 2560),
            num_layers=model_cfg.get("num_layers", 20),
            max_seq_len=model_cfg.get("max_seq_len", 2048),
            pad_token_id=model_cfg.get("pad_token_id", 0),
            bos_token_id=model_cfg.get("bos_token_id", 1),
            eos_token_id=model_cfg.get("eos_token_id", 2),
        )
        model = HavocModel(config)

    # Load weights
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {param_count:.2f}B parameters")

    return model, config


# =============================================================================
# GENERATION
# =============================================================================
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt with safe sampling."""
    model.eval()

    # Encode prompt
    if hasattr(tokenizer, "encode"):
        # HavocTokenizer
        input_ids = tokenizer.encode(prompt)
    else:
        # HuggingFace tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Get EOS token ID
    eos_id = getattr(tokenizer, "eos_token_id", 2)

    # Generation loop
    generated = input_tensor.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass - handle tuple return (logits, past_kv)
            outputs = model(generated)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Get last token logits
            next_token_logits = logits[:, -1, :].float()  # Cast to float32 for stability

            # Check for NaN/Inf
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("Warning: NaN/Inf in logits, stopping generation")
                break

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)

            # Guard against all -inf (shouldn't happen but safety check)
            if probs.sum() == 0 or torch.isnan(probs).any():
                print("Warning: Invalid probability distribution, using greedy")
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Append token
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS
            if next_token.item() == eos_id:
                break

    # Decode
    output_ids = generated[0].tolist()
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(output_ids)
    else:
        return tokenizer.decode(output_ids, skip_special_tokens=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HAVOC Interactive Chat")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k sampling threshold",
    )

    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'unknown')}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Check vocab compatibility
    model_vocab = getattr(config, "vocab_size", None)
    tok_vocab = getattr(tokenizer, "vocab_size", None)
    if model_vocab and tok_vocab and model_vocab != tok_vocab:
        print(f"WARNING: Vocab mismatch! Model={model_vocab}, Tokenizer={tok_vocab}")

    print("\n" + "=" * 50)
    print("HAVOC Chat")
    print("=" * 50)
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 50)
    print("Type your message and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Shutting down.")
            break

        # Generate response
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=user_input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )

        print(f"HAVOC: {response}\n")


if __name__ == "__main__":
    main()
