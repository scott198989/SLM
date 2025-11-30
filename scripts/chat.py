"""
HAVOC interactive chat script (improved v2).
Adds:
- System prompt
- Conversation history
- Honest fallback ("I don't know based on my training data.")
- Role formatting
- Greeting handling
- Basic gibberish / SMILES filter for non-chem questions
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional, List

import torch

from havoc_core.config import HavocConfig, AttentionConfig, MLPConfig
from havoc_core.model.transformer import HavocModel
from havoc_core.tokenizer.tokenizer import load_tokenizer


# -----------------------------------------------------
# System Prompt
# -----------------------------------------------------
SYSTEM_PROMPT = (
    "System:\n"
    "You are HAVOC, a domain-focused chemistry and science assistant.\n"
    "You answer questions directly, concisely, and honestly.\n"
    "You MUST NOT fabricate mechanisms, spectra, reactions, or structures if unsure.\n"
    "If you do not know, respond exactly with:\n"
    "\"I don't know based on my training data.\"\n"
    "Do NOT continue synthetic procedures unless explicitly asked.\n"
    "When the user asks a question, respond to the question.\n"
    "Use plain English unless chemical notation is necessary.\n"
    "Stay in assistant mode at all times.\n"
)


# -----------------------------------------------------
# Build the full prompt with history
# -----------------------------------------------------
def build_prompt(history: List[str]) -> str:
    """
    history = [
        "User: ...",
        "HAVOC: ...",
        "User: ...",
    ]
    """
    return SYSTEM_PROMPT + "\n" + "\n".join(history) + "\nHAVOC: "


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

    weights_path = os.path.join(checkpoint_dir, "model.pt")
    print(f"Loading PyTorch weights: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()
    return model


# -----------------------------------------------------
# Small helpers for output sanity
# -----------------------------------------------------
def _looks_like_smiles_gibberish(text: str) -> bool:
    # Very crude: catches atom-mapped / bracket-heavy junk
    smiles_markers = ["[CH", "[C:", "[N:", "[O:", "m/z", "δ ", "SMILES"]
    if any(tok in text for tok in smiles_markers):
        return True

    # Bracket density as a rough signal
    bracket_count = text.count("[") + text.count("]")
    ratio = bracket_count / max(1, len(text))
    return ratio > 0.12


def _user_is_greeting(user_msg: str) -> bool:
    msg = user_msg.lower().strip()
    greeting_triggers = ("hello", "hi", "hey", "yo", "sup", "good morning", "good afternoon", "good evening")
    return any(msg.startswith(g) for g in greeting_triggers) or msg in {"hello", "hi", "hey", "yo", "sup"}


def _user_requested_chem(user_msg: str) -> bool:
    """Does the user clearly want chemistry / SMILES / spectra?"""
    msg = user_msg.lower()
    chem_tokens = [
        "smiles",
        "δ",
        "nmr",
        "logp",
        "reaction",
        "mechanism",
        "naBH4".lower(),
        "ester",
        "amide",
        "alkoxide",
        "compound",
        "synthesis",
        "[",  # raw SMILES / atom-mapped
    ]
    return any(tok in msg for tok in chem_tokens)


# -----------------------------------------------------
# Generate Reply
# -----------------------------------------------------
def generate_reply(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    last_user_msg: str,
) -> str:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    new_tokens = out[0, ids.shape[1]:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Empty / nothing
    if not text:
        return "I don't know based on my training data."

    # If user just greeted and we got chem-gibberish → override with a normal greeting
    if _user_is_greeting(last_user_msg) and _looks_like_smiles_gibberish(text):
        return "Hey, I'm HAVOC. I'm better at chemistry than small talk, but I'm here. What do you want to work on?"

    # If user did NOT ask for chemistry, but output is SMILES/NMR gobble:
    if not _user_requested_chem(last_user_msg) and _looks_like_smiles_gibberish(text):
        return "I don't know based on my training data."

    # Additional simple bad-markers for early cutoff
    gibberish_markers = ["Â", "Â°", "old mixture"]
    if any(x in text for x in gibberish_markers) and not _user_requested_chem(last_user_msg):
        return "I don't know based on my training data."

    return text


# -----------------------------------------------------
# Chat Loop
# -----------------------------------------------------
def chat_loop(model, tokenizer, device, max_new_tokens, temperature):
    print("Type 'quit' to exit.\n")

    history: List[str] = []

    while True:
        try:
            msg = input("You: ").strip()
        except EOFError:
            print("\nEOF reached. Exiting.")
            break

        if msg.lower() in {"quit", "exit", "q"}:
            print("Goodnight.")
            break

        # Add user message to history
        history.append(f"User: {msg}")

        # Trim history so we don't blow context
        if len(history) > 20:
            history = history[-20:]

        # Build full prompt
        full_prompt = build_prompt(history)

        # Generate reply
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            last_user_msg=msg,
        )

        # Append reply to history
        history.append(f"HAVOC: {reply}")

        print(f"HAVOC: {reply}\n")


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chat with HAVOC checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/SLM/checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default="/workspace/SLM/artifacts/tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device(args.device)

    chk = Path(args.checkpoint_dir)

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
