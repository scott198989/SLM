import torch
import readline
import json
import sys
import time

from havoc_core.model.transformer import HavocModel
from havoc_core.tokenizer.tokenizer import HavocTokenizer
from havoc_core.config import HavocConfig
from havoc_core.attention import AttentionConfig
from havoc_core.mlp import MLPConfig

CHECKPOINT = "checkpoints/havoc_phase1_sft/checkpoint_step_50000"
TOKENIZER_PATH = "artifacts/tokenizer"

print("\n=== HAVOC CHAT INTERFACE LOADED ===\n")

# -------------------------------------------------------
# Load tokenizer
# -------------------------------------------------------
tok = HavocTokenizer(TOKENIZER_PATH)

# -------------------------------------------------------
# Load config
# -------------------------------------------------------
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

# -------------------------------------------------------
# Load model weights
# -------------------------------------------------------
model = HavocModel(model_cfg)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e9:.2f}B")

state = torch.load(f"{CHECKPOINT}/model.pt", map_location="cuda")
model.load_state_dict(state)
model.to("cuda")
model.eval()

print("Model loaded successfully!\n")

# -------------------------
# Chat template + history
# -------------------------

history = []

def build_prompt(history, user_msg):
    prompt = ""
    for turn in history:
        prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt += f"User: {user_msg}\nAssistant:"
    return prompt

# -------------------------------------------------------
# Generate
# -------------------------------------------------------
def generate(user_msg):
    global history

    prompt = build_prompt(history, user_msg)

    ids = tok.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device="cuda")

    # Your HavocModel.generate signature only supports:
    # (prompt_ids, max_new_tokens, temperature, tokenizer)
    output = model.generate(
        prompt_ids=ids,
        max_new_tokens=200,
        temperature=0.9,
        tokenizer=tok
    )

    out_ids = output[0].tolist()
    text = tok.decode(out_ids)

    # strip prompt from output
    reply = text[len(prompt):].strip()

    # save chat history
    history.append({"user": user_msg, "assistant": reply})

    return reply

# -------------------------------------------------------
# Chat loop
# -------------------------------------------------------
while True:
    try:
        msg = input("You: ").strip()
        if msg.lower() in ("exit", "quit"):
            sys.exit(0)

        reply = generate(msg)
        print(f"HAVOC: {reply}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
