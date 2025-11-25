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

    # Load weights from model.pt
    weights_path = os.path.join(checkpoint_dir, "model.pt")
    print(f"Loading PyTorch weights: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()
    return model


# -----------------------------------------------------
# Generate reply — fixed **without tokenizer argument**
# -----------------------------------------------------
def generate_reply(model, tokenizer, conversation, system_prompt, max_new_tokens, temperature, device):
    full_prompt = system_prompt + "\n" + "\n".join(conversation) + "\nHAVOC:"

    input_ids = tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        # model.generate() DOES NOT support tokenizer → so we call it clean
        output = model.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # Extract new tokens
    new_tokens = output[0, input_ids.shape[1]:].tolist()
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Last line of defense against "." spam
    if raw.strip() == ".":
        raw = "(mumbles and shrugs)"

    if raw == "":
        raw = "(mumbles unintelligibly)"

    return raw


# -----------------------------------------------------
# Chat Loop
# -----------------------------------------------------
def chat_loop(model, tokenizer, device, max_new_tokens, temperature):
    system_prompt = (
        "SYSTEM: You are HAVOC, a small experimental model created by Scott Tuschl. "
        "You respond in short, weird, semi-coherent sentences. DO NOT simply repeat Scott."
    )

    conversation = []
    print("Type 'quit' to exit.\n")

    while True:
        try:
            msg = input("You: ").strip()
        except EOFError:
            print("\nExiting.")
            break

        if msg.lower() in {"quit", "exit", "q"}:
            print("Goodnight.")
            break

        if not msg:
            continue

        conversation.append(f"You: {msg}")

        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            conversation=conversation,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

        print(f"HAVOC: {reply}\n")
        conversation.append(f"HAVOC: {reply}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chat with HAVOC checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device(args.device)

    latest = find_latest_checkpoint(args.checkpoint_dir)
    print(f"Using checkpoint: {latest}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(str(latest), device=device)

    chat_loop(model, tokenizer, device, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
