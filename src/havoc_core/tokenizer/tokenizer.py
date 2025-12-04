import os
import json
from typing import List, Dict, Optional

import sentencepiece as spm


class HavocTokenizer:
    """
    Custom tokenizer for HAVOC models using SentencePiece BPE.
    Provides encode/decode and exposes token_id fields expected by inference code.
    """

    def __init__(self, model_path: str):
        spm_path = os.path.join(model_path, "tokenizer.model")
        vocab_path = os.path.join(model_path, "vocab.json")

        if not os.path.exists(spm_path):
            raise FileNotFoundError(f"SentencePiece model not found at: {spm_path}")

        self.sp = spm.SentencePieceProcessor(model_file=spm_path)

        # Load vocab.json if available
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                vocab_data = json.load(f)

            self.id_to_token = vocab_data.get("id_to_token", {})
            self.token_to_id = vocab_data.get("token_to_id", {})
        else:
            # Construct basic id→token mapping
            self.id_to_token = {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}
            self.token_to_id = {v: k for k, v in self.id_to_token.items()}

        # Required for inference engine & model.generate()
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        # Consistency flags
        self.vocab_size = self.sp.get_piece_size()

    # -----------------------------
    # Core encode/decode
    # -----------------------------
    def __call__(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Make tokenizer callable for compatibility with dataset.
        Encode text into token IDs with optional BOS/EOS tokens.
        """
        token_ids = self.sp.encode(text, out_type=int)
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        return [self.bos_token_id] + self.sp.encode(text, out_type=int) + [self.eos_token_id]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs into text."""
        # Remove BOS / PAD
        cleaned = []
        for t in ids:
            if t in (self.bos_token_id, self.pad_token_id):
                continue
            if t == self.eos_token_id:
                break
            cleaned.append(t)

        return self.sp.decode(cleaned)

    # -----------------------------
    # HuggingFace-style helpers
    # -----------------------------
    @property
    def vocab(self) -> Dict[str, int]:
        return self.token_to_id

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, "<unk>") for i in ids]


# ============================================================
# Loader used by InferenceEngine
# ============================================================

def load_tokenizer(path: str) -> HavocTokenizer:
    """
    Loads the tokenizer from artifact directory.
    This is what InferenceEngine calls.
    """
    return HavocTokenizer(path)
