from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import sentencepiece as spm


class HavocTokenizer:
    """
    Wrapper for SentencePiece tokenizer with HAVOC-specific functionality.

    Provides encoding/decoding with support for:
    - SRS reasoning stage markers
    - DSL boundary markers
    - Tool invocation markers
    - Engineering symbols and units
    - Math symbols and Greek letters
    """

    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Initialize tokenizer from trained model.

        Args:
            model_path: Path to .model file or directory containing tokenizer.model
            metadata_path: Optional path to tokenizer_metadata.json
        """
        # Handle directory or file path
        model_path_obj = Path(model_path)
        if model_path_obj.is_dir():
            self.model_path = model_path_obj / "tokenizer.model"
            self.metadata_path = metadata_path or (model_path_obj / "tokenizer_metadata.json")
        else:
            self.model_path = model_path_obj
            self.metadata_path = metadata_path

        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))

        # Load metadata if available
        self.metadata: Optional[Dict] = None
        if self.metadata_path and Path(self.metadata_path).exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Special token IDs
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()

        # Build special token mappings
        self._build_special_token_map()

    def _build_special_token_map(self) -> None:
        """Build mappings for special tokens."""
        self.special_tokens: Dict[str, int] = {}
        self.special_ids: Dict[int, str] = {}

        # Core tokens
        self.special_tokens["<pad>"] = self.pad_id
        self.special_tokens["<bos>"] = self.bos_id
        self.special_tokens["<eos>"] = self.eos_id
        self.special_tokens["<unk>"] = self.unk_id

        self.special_ids[self.pad_id] = "<pad>"
        self.special_ids[self.bos_id] = "<bos>"
        self.special_ids[self.eos_id] = "<eos>"
        self.special_ids[self.unk_id] = "<unk>"

        # If we have metadata, register all special tokens
        if self.metadata and "special_tokens" in self.metadata:
            for token in self.metadata["special_tokens"]:
                token_id = self.sp.PieceToId(token)
                if token_id != self.unk_id:  # Only if token exists in vocab
                    self.special_tokens[token] = token_id
                    self.special_ids[token_id] = token

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.GetPieceSize()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        out_type: type = int,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Whether to add BOS token at start
            add_eos: Whether to add EOS token at end
            out_type: Output type (int or str)

        Returns:
            List of token IDs
        """
        token_ids = self.sp.Encode(text, out_type=out_type)

        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs (single list or batch)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text (single string or list of strings)
        """
        # Check if batch (list of lists)
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]

        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [
                tid for tid in token_ids
                if tid not in (self.pad_id, self.bos_id, self.eos_id)
            ]

        return self.sp.Decode(token_ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_bos, add_eos) for text in texts]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into pieces (for debugging).

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to IDs."""
        return [self.sp.PieceToId(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to strings."""
        return [self.sp.IdToPiece(token_id) for token_id in ids]

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get ID for a special token."""
        return self.special_tokens.get(token)

    def get_srs_token_ids(self) -> Dict[str, int]:
        """Get all SRS reasoning stage token IDs."""
        srs_tokens = [
            "<SRS_MODE>",
            "<SRS_GROUND>",
            "<SRS_PLAN>",
            "<SRS_EXECUTE>",
            "<SRS_ARGUE>",
            "<SRS_ARBITER>",
            "<SRS_AUDIT>",
            "<SRS_ANSWER>",
        ]
        return {
            token: self.special_tokens[token]
            for token in srs_tokens
            if token in self.special_tokens
        }

    def get_dsl_token_ids(self) -> Dict[str, int]:
        """Get DSL boundary marker token IDs."""
        dsl_tokens = ["<DSL_BEGIN>", "<DSL_END>"]
        return {
            token: self.special_tokens[token]
            for token in dsl_tokens
            if token in self.special_tokens
        }

    def get_tool_token_ids(self) -> Dict[str, int]:
        """Get tool invocation marker token IDs."""
        tool_tokens = ["<TOOL_MATH>", "<TOOL_STATS>"]
        return {
            token: self.special_tokens[token]
            for token in tool_tokens
            if token in self.special_tokens
        }

    def __call__(
        self,
        text: Union[str, List[str]],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """
        Make tokenizer callable for easier use.

        Args:
            text: Input text or list of texts
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token

        Returns:
            Token IDs (single list or batch)
        """
        if isinstance(text, str):
            return self.encode(text, add_bos, add_eos)
        else:
            return self.encode_batch(text, add_bos, add_eos)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HavocTokenizer(vocab_size={self.vocab_size}, "
            f"model_path={self.model_path})"
        )


def load_tokenizer(path: str) -> HavocTokenizer:
    """
    Load a trained HAVOC tokenizer.

    Args:
        path: Path to tokenizer directory or .model file

    Returns:
        HavocTokenizer instance
    """
    return HavocTokenizer(path)
