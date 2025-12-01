#!/usr/bin/env python3
"""Quick test script for HAVOC chat functionality"""

import torch
import json
from havoc_core.model.transformer import HavocModel
from havoc_core.tokenizer.tokenizer import HavocTokenizer
from havoc_core.config import HavocConfig
from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig

CHECKPOINT = "checkpoints/havoc_phase1_sft/checkpoint_step_50000"
TOKENIZER_PATH = "artifacts/tokenizer"

print("\n=== TESTING HAVOC CHAT ===\n")

# Load tokenizer
tok = HavocTokenizer(TOKENIZER_PATH)

# Load config from checkpoint
with open(f"{CHECKPOINT}/config.json", "r") as f:
    cfg_raw = json.load(f)["model_config"]

attn_cfg = AttentionConfig(
    num_heads=cfg_raw["attention"]["num_heads"],
    head_dim=cfg_raw["attention"]["head_dim"],
    num_kv_heads=cfg_raw["attention"]["num_kv_heads"],
    dropout=cfg_raw["attention"]["dropout"],
    rotary_dim=cfg_raw["attention"]["rotary_dim"],
    rope_theta=cfg_raw["attention"]["rope_theta"],
    bias=cfg_raw["attention"]["bias"],
)

mlp_cfg = MLPConfig(
    hidden_dim=cfg_raw["mlp"]["hidden_dim"],
    activation=cfg_raw["mlp"]["activation"],
    dropout=cfg_raw["mlp"]["dropout"],
)

model_cfg = HavocConfig(
    vocab_size=cfg_raw["vocab_size"],
    d_model=cfg_raw["d_model"],
    num_layers=cfg_raw["num_layers"],
    max_seq_len=cfg_raw["max_seq_len"],
    attention=attn_cfg,
    mlp=mlp_cfg,
    dropout=cfg_raw["dropout"],
    layer_norm_eps=cfg_raw["layer_norm_eps"],
    initializer_range=cfg_raw["initializer_range"],
    pad_token_id=cfg_raw["pad_token_id"],
    bos_token_id=cfg_raw["bos_token_id"],
    eos_token_id=cfg_raw["eos_token_id"],
)

# Load model
print("Loading model...")
model = HavocModel(model_cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e9:.2f}B")

state = torch.load(f"{CHECKPOINT}/model.pt", map_location="cuda")
model.load_state_dict(state)
model.to("cuda")
model.eval()
print("Model loaded successfully!\n")

# Test generation
def test_generate(prompt):
    print(f"Prompt: {prompt}")
    ids = tok.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device="cuda")

    output = model.generate(
        prompt_ids=ids,
        max_new_tokens=64,
        temperature=0.8,
        tokenizer=tok
    )

    out_ids = output[0].tolist()
    result = tok.decode(out_ids)
    print(f"Response: {result}\n")
    return result

# Run test cases
test_prompts = [
    "What is 2+2?",
    "Explain what is a transformer model.",
    "Calculate the mean of [1, 2, 3, 4, 5]."
]

for prompt in test_prompts:
    try:
        test_generate(prompt)
    except Exception as e:
        print(f"Error: {e}\n")

print("=== TEST COMPLETE ===")
