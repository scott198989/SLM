import torch
from pathlib import Path
from havoc_core.model.prime_model import HavocPrimeModel
from havoc_core.config_7b import Havoc7BConfig
from transformers import PreTrainedTokenizerFast

# ================================================
# CONFIGURE THESE PATHS
# ================================================
CHECKPOINT_DIR = "/workspace/SLM/checkpoints/havoc_phase0_complete/checkpoint_interrupted"
TOKENIZER_PATH = "/workspace/SLM/artifacts/tokenizer"

# ================================================
# LOAD TOKENIZER
# ================================================
print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

# ================================================
# LOAD MODEL CONFIG + MODEL
# ================================================
print(f"Loading model config from: {CHECKPOINT_DIR}")

config_path = Path(CHECKPOINT_DIR) / "config.json"
config = Havoc7BConfig.from_json_file(str(config_path))

print("Initializing model...")
model = HavocPrimeModel(config)

# ================================================
# LOAD MODEL WEIGHTS
# ================================================
weights_path = Path(CHECKPOINT_DIR) / "model.pt"
print(f"Loading model weights from: {weights_path}")

state_dict = torch.load(weights_path, map_location="cpu")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("\nLoad report:")
print("  Missing keys:", missing)
print("  Unexpected keys:", unexpected)

model.eval()
model.to("cuda")

print("\n✓ HAVOC model loaded and ready.\n")

# ================================================
# GENERATION FUNCTION
# ================================================
def generate(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================================================
# INTERACTIVE CHAT LOOP
# ================================================
print("Type your messages to talk to HAVOC.")
print("Type 'exit' to quit.")
print("---------------------------------------------------")

while True:
    user_in = input("\nYou: ")
    if user_in.lower().strip() in ["exit", "quit"]:
        break

    response = generate(user_in)
    print(f"HAVOC: {response}")

print("Shutting down.")
