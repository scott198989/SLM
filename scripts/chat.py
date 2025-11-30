"""
HAVOC interactive chat script.
Model.generate() DOES NOT accept tokenizer → handled here instead.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch

from havoc_core.config import HavocConfig, AttentionConfig, MLPConfig
from havoc_core.model.transformer import HavocModel
from havoc_core.tokenizer.tokenizer import load_tokenizer


# -----------------------------------------------------
# Find Latest Checkpoint
# -----------------------------------------------------
def find_latest_checkpoint(base_dir: str) -> Optional[Path]:
    base = Path(base_dir)
    if not base.exists():
        return None

    checkpoints = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


# -----------------------------------------------------
# Load Model + Rebuild Config
# -----------------------------------------------------
def load_model(checkpoint_dir: str, device="cuda"):
    print(f"Loading model from: {checkpoint_dir}")

    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        raw = json.load(f)

    m = raw["model_config"]

    attention_cfg = AttentionConfig(**m["attention"])
    mlp_cfg = MLPConfig(**m["mlp"])

    model_cfg = HavocConfig(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        num_layers=m["num_layers"],
        max_seq_len=m["max_seq_len"],
        attention=attention_cfg,
        mlp=mlp_cfg,
        dropout=m.get("dropout", 0.0),
        layer_norm_eps=m.get("layer_norm_eps", 1e-5),
        initializer_range=m.get("initializer_range", 0.02),
        pad_token_id=m.get("pad_token_id", 0),
        bos_token_id=m.get("bos_token_id", 1),
        eos_token_id=m.get("eos_token_id", 2),
    )

    model = HavocModel(model_cfg).to(device)

    # Load weights
    weights_path = os.path.join(checkpoint_dir, "model.pt")
    print(f"Loading PyTorch weights: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()
    return model


# -----------------------------------------------------
# Chat loop
# -----------------------------------------------------
def chat_loop(model, tokenizer, device, max_new_tokens, temperature):
    print("Type 'quit' to exit.\n")

    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"quit", "exit", "q"}:
            print("Goodnight.")
            break

        ids = tokenizer.encode(msg, add_bos=True, add_eos=False)
        ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                prompt_ids=ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        new_tokens = out[0, ids.shape[1]:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if text == "":
            text = "(mumbles)"

        print(f"HAVOC: {text}\n")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chat with HAVOC checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/SLM/checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default="/workspace/SLM/artifacts/tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- FIXED: allow direct checkpoint folder or directory containing multiple checkpoints
    chk = Path(args.checkpoint-dir)

    if chk.exists():
        if (chk / "model.pt").exists() and (chk / "config.json").exists():
            latest = chk
        else:
            latest = find_latest_checkpoint(str(chk))
    else:
        latest = None

    if latest is None:
        raise FileNotFoundError(f"No valid checkpoint found under: {args.checkpoint_dir}")

    print(f"Using checkpoint: {latest}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(str(latest), device=device)

    chat_loop(model, tokenizer, device, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
