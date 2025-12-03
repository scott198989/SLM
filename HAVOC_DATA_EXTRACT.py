import os
import json
import sentencepiece as spm

# Path to your tokenizer on Windows (use the .model file)
TOKENIZER_PATH = r"C:\Users\ScottT\SLM\artifacts\tokenizer\tokenizer.model"

# Folder containing ALL your data
DATA_FOLDER = r"C:\Users\ScottT\Desktop\HAVOC_DATA"

# Load SentencePiece tokenizer
tok = spm.SentencePieceProcessor()
tok.Load(TOKENIZER_PATH)

total_tokens = 0
total_samples = 0

def process_jsonl(path):
    global total_tokens, total_samples
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue

            text = obj.get("text") or obj.get("prompt") or ""
            # SentencePiece encode returns a list of token IDs
            ids = tok.EncodeAsIds(text)
            total_tokens += len(ids)
            total_samples += 1

for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith(".jsonl"):
            full_path = os.path.join(root, file)
            print(f"Processing: {full_path}")
            process_jsonl(full_path)

print("\n===================================")
print("TOTAL TOKEN COUNT:", total_tokens)
print("TOTAL SAMPLES:", total_samples)
print("AVERAGE TOKENS PER SAMPLE:", total_tokens / max(1, total_samples))
