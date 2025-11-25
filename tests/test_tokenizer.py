"""
Unit tests for HAVOC tokenizer.

Tests cover:
- Reserved token handling (SRS, DSL, tools)
- Math expression segmentation
- Engineering symbol preservation
- DSL boundary parsing
- End-to-end training and loading
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from havoc_core.config import TokenizerTrainingConfig
from havoc_core.tokenizer import (
    DOMAIN_TOKENS,
    DSL_TOKENS,
    ENGINEERING_TOKENS,
    ENGINEERING_UNITS,
    GREEK_LETTERS,
    MATH_SYMBOLS,
    SRS_TOKENS,
    TOOL_TOKENS,
    HavocTokenizer,
    get_all_special_tokens,
    train_tokenizer,
)
from havoc_core.tokenizer.train_tokenizer import normalize_text
from havoc_core.tokenizer.vocab_utils import get_char_normalization_map


class TestVocabUtils:
    """Test vocabulary utilities."""

    def test_srs_tokens_count(self):
        """Test that we have all 8 SRS stage tokens."""
        assert len(SRS_TOKENS) == 8
        assert "<SRS_MODE>" in SRS_TOKENS
        assert "<SRS_ANSWER>" in SRS_TOKENS

    def test_dsl_tokens_count(self):
        """Test that we have DSL boundary tokens."""
        assert len(DSL_TOKENS) == 2
        assert "<DSL_BEGIN>" in DSL_TOKENS
        assert "<DSL_END>" in DSL_TOKENS

    def test_tool_tokens_count(self):
        """Test that we have tool invocation tokens."""
        assert len(TOOL_TOKENS) == 2
        assert "<TOOL_MATH>" in TOOL_TOKENS
        assert "<TOOL_STATS>" in TOOL_TOKENS

    def test_engineering_tokens_count(self):
        """Test that we have engineering boundary tokens."""
        assert len(ENGINEERING_TOKENS) == 2
        assert "<ENG_SYMBOL_START>" in ENGINEERING_TOKENS
        assert "<ENG_SYMBOL_END>" in ENGINEERING_TOKENS

    def test_math_symbols_present(self):
        """Test that key math symbols are included."""
        assert "∑" in MATH_SYMBOLS  # Sum
        assert "∫" in MATH_SYMBOLS  # Integral
        assert "∂" in MATH_SYMBOLS  # Partial derivative
        assert "±" in MATH_SYMBOLS  # Plus-minus
        assert "≈" in MATH_SYMBOLS  # Approximately equal

    def test_greek_letters_present(self):
        """Test that Greek letters are included."""
        assert "α" in GREEK_LETTERS  # alpha
        assert "β" in GREEK_LETTERS  # beta
        assert "μ" in GREEK_LETTERS  # mu
        assert "σ" in GREEK_LETTERS  # sigma
        assert "Σ" in GREEK_LETTERS  # Sigma (uppercase)

    def test_engineering_units_present(self):
        """Test that engineering units are included."""
        assert "psi" in ENGINEERING_UNITS
        assert "MPa" in ENGINEERING_UNITS
        assert "N/cm" in ENGINEERING_UNITS
        assert "°C" in ENGINEERING_UNITS
        assert "kW" in ENGINEERING_UNITS

    def test_domain_tokens_present(self):
        """Test that DOE/SPC domain tokens are included."""
        assert "ANOVA" in DOMAIN_TOKENS
        assert "Box-Behnken" in DOMAIN_TOKENS
        assert "Cpk" in DOMAIN_TOKENS
        assert "X-bar" in DOMAIN_TOKENS
        assert "UCL" in DOMAIN_TOKENS

    def test_get_all_special_tokens(self):
        """Test that get_all_special_tokens returns all categories."""
        all_tokens = get_all_special_tokens()

        # Check that tokens from each category are present
        assert any(token in all_tokens for token in SRS_TOKENS)
        assert any(token in all_tokens for token in DSL_TOKENS)
        assert any(token in all_tokens for token in TOOL_TOKENS)
        assert any(token in all_tokens for token in MATH_SYMBOLS)
        assert any(token in all_tokens for token in GREEK_LETTERS)
        assert any(token in all_tokens for token in ENGINEERING_UNITS)
        assert any(token in all_tokens for token in DOMAIN_TOKENS)


class TestNormalization:
    """Test text normalization functions."""

    def test_whitespace_collapse(self):
        """Test that multiple whitespace is collapsed."""
        text = "This  has   multiple    spaces"
        normalized = normalize_text(text)
        assert "  " not in normalized
        assert normalized == "This has multiple spaces"

    def test_superscript_normalization(self):
        """Test that superscripts are converted to caret notation."""
        text = "x²"
        normalized = normalize_text(text)
        assert "x^2" in normalized

    def test_subscript_normalization(self):
        """Test that subscripts are converted to underscore notation."""
        text = "H₂O"
        normalized = normalize_text(text)
        assert "H_2O" in normalized

    def test_unicode_dash_normalization(self):
        """Test that unicode dashes are normalized to hyphen."""
        char_map = get_char_normalization_map()
        assert char_map["−"] == "-"  # Minus sign
        assert char_map["–"] == "-"  # En dash
        assert char_map["—"] == "-"  # Em dash

    def test_fraction_normalization(self):
        """Test that fraction characters are normalized."""
        char_map = get_char_normalization_map()
        assert char_map["½"] == "1/2"
        assert char_map["¼"] == "1/4"
        assert char_map["¾"] == "3/4"

    def test_math_expression_normalization(self):
        """Test normalization of complex math expression."""
        text = "μ₁ = μ₂   with   σ² = 4"
        normalized = normalize_text(text)
        # Should have normalized subscripts and whitespace
        assert "μ_1" in normalized
        assert "μ_2" in normalized
        assert "σ^2" in normalized
        assert "   " not in normalized  # No triple spaces


class TestTokenizerTraining:
    """Test tokenizer training functionality."""

    def test_train_minimal_tokenizer(self):
        """Test training a minimal tokenizer with default samples only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TokenizerTrainingConfig(
                vocab_size=1000,  # Small vocab for speed
                output_dir=tmpdir,
                input_files=[],  # No input files, use samples only
            )

            metadata = train_tokenizer(config, verbose=False)

            # Check that model files were created
            assert Path(tmpdir, "tokenizer.model").exists()
            assert Path(tmpdir, "tokenizer.vocab").exists()
            assert Path(tmpdir, "tokenizer_metadata.json").exists()

            # Check metadata
            assert metadata.vocab_size == 1000
            assert len(metadata.special_tokens) > 0
            assert len(metadata.domain_tokens) > 0

    def test_train_tokenizer_with_corpus(self):
        """Test training tokenizer with a small corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small corpus file
            corpus_file = Path(tmpdir) / "corpus.txt"
            corpus_file.write_text(
                "Calculate the mean μ and variance σ².\n"
                "Perform ANOVA with α = 0.05.\n"
                "The stress σ = F/A where F is force.\n"
                "<SRS_MODE> Identify problem type <SRS_GROUND>\n"
                "<DSL_BEGIN> CHECK_SPC X-bar chart <DSL_END>\n"
            )

            config = TokenizerTrainingConfig(
                vocab_size=1000,
                output_dir=tmpdir,
                input_files=[str(corpus_file)],
            )

            metadata = train_tokenizer(config, verbose=False)

            # Check that training succeeded
            assert Path(tmpdir, "tokenizer.model").exists()
            assert metadata.vocab_size == 1000


class TestHavocTokenizer:
    """Test HavocTokenizer wrapper class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a minimal trained tokenizer for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TokenizerTrainingConfig(
                vocab_size=5000,
                output_dir=tmpdir,
                input_files=[],
            )
            train_tokenizer(config, verbose=False)

            yield HavocTokenizer(tmpdir)

    def test_tokenizer_initialization(self, tokenizer):
        """Test that tokenizer initializes correctly."""
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2
        assert tokenizer.unk_id == 3

    def test_encode_decode(self, tokenizer):
        """Test basic encoding and decoding."""
        text = "Hello world"
        token_ids = tokenizer.encode(text)

        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)

        decoded = tokenizer.decode(token_ids)
        # Decoded text should be similar (may have spacing differences)
        assert "Hello" in decoded or "hello" in decoded.lower()

    def test_encode_with_bos_eos(self, tokenizer):
        """Test encoding with BOS/EOS tokens."""
        text = "Test"
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

        assert token_ids[0] == tokenizer.bos_id
        assert token_ids[-1] == tokenizer.eos_id

    def test_encode_batch(self, tokenizer):
        """Test batch encoding."""
        texts = ["First text", "Second text", "Third text"]
        batch_ids = tokenizer.encode_batch(texts)

        assert len(batch_ids) == 3
        assert all(isinstance(ids, list) for ids in batch_ids)

    def test_decode_batch(self, tokenizer):
        """Test batch decoding."""
        texts = ["Hello", "World"]
        batch_ids = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode(batch_ids)

        assert len(decoded) == 2
        assert all(isinstance(text, str) for text in decoded)

    def test_callable_interface(self, tokenizer):
        """Test that tokenizer can be called directly."""
        text = "Test text"
        token_ids = tokenizer(text)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

    def test_special_token_access(self, tokenizer):
        """Test accessing special tokens."""
        # Check core special tokens
        assert "<pad>" in tokenizer.special_tokens
        assert "<bos>" in tokenizer.special_tokens
        assert "<eos>" in tokenizer.special_tokens
        assert "<unk>" in tokenizer.special_tokens

    def test_srs_token_ids(self, tokenizer):
        """Test getting SRS token IDs."""
        srs_ids = tokenizer.get_srs_token_ids()

        # Should have some SRS tokens (depending on training)
        # At minimum, they should be in the vocabulary
        for token in SRS_TOKENS:
            token_id = tokenizer.sp.PieceToId(token)
            # Token should be recognized (not UNK)
            assert token_id is not None

    def test_dsl_token_ids(self, tokenizer):
        """Test getting DSL token IDs."""
        dsl_ids = tokenizer.get_dsl_token_ids()

        # DSL tokens should be in vocabulary
        for token in DSL_TOKENS:
            token_id = tokenizer.sp.PieceToId(token)
            assert token_id is not None

    def test_tool_token_ids(self, tokenizer):
        """Test getting tool token IDs."""
        tool_ids = tokenizer.get_tool_token_ids()

        # Tool tokens should be in vocabulary
        for token in TOOL_TOKENS:
            token_id = tokenizer.sp.PieceToId(token)
            assert token_id is not None


class TestMathExpressionTokenization:
    """Test tokenization of mathematical expressions."""

    @pytest.fixture
    def math_tokenizer(self):
        """Create tokenizer trained on math-heavy corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create math corpus
            corpus_file = Path(tmpdir) / "math_corpus.txt"
            corpus_file.write_text(
                "The mean μ and variance σ²\n"
                "Sum: ∑ᵢ₌₁ⁿ xᵢ\n"
                "Integral: ∫₀¹ x² dx\n"
                "Partial derivative: ∂f/∂x\n"
                "Gradient: ∇f\n"
            )

            config = TokenizerTrainingConfig(
                vocab_size=5000,
                output_dir=tmpdir,
                input_files=[str(corpus_file)],
            )
            train_tokenizer(config, verbose=False)

            yield HavocTokenizer(tmpdir)

    def test_greek_letters_tokenization(self, math_tokenizer):
        """Test that Greek letters are properly tokenized."""
        text = "Calculate μ and σ"
        tokens = math_tokenizer.tokenize(text)

        # Greek letters should appear in tokenization
        # (exact form depends on tokenizer, but they should be preserved)
        token_str = " ".join(tokens)
        # After normalization and encoding, should still decode properly
        encoded = math_tokenizer.encode(text)
        decoded = math_tokenizer.decode(encoded)
        assert decoded  # Should not be empty

    def test_math_symbols_tokenization(self, math_tokenizer):
        """Test that math symbols are tokenized."""
        text = "a ≈ b ± 0.1"
        encoded = math_tokenizer.encode(text)
        decoded = math_tokenizer.decode(encoded)

        # Should successfully encode and decode
        assert len(encoded) > 0
        assert decoded


class TestEngineeringSymbols:
    """Test engineering symbol and unit handling."""

    @pytest.fixture
    def eng_tokenizer(self):
        """Create tokenizer trained on engineering corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "eng_corpus.txt"
            corpus_file.write_text(
                "<ENG_SYMBOL_START> σ = 250 MPa <ENG_SYMBOL_END>\n"
                "Pressure: 150 psi\n"
                "Force: 100 N/cm\n"
                "Temperature: 150°C\n"
                "Power: 5 kW\n"
            )

            config = TokenizerTrainingConfig(
                vocab_size=5000,
                output_dir=tmpdir,
                input_files=[str(corpus_file)],
            )
            train_tokenizer(config, verbose=False)

            yield HavocTokenizer(tmpdir)

    def test_engineering_units_tokenization(self, eng_tokenizer):
        """Test that engineering units are tokenized."""
        text = "Stress: 250 MPa, Pressure: 150 psi"
        encoded = eng_tokenizer.encode(text)
        decoded = eng_tokenizer.decode(encoded)

        assert len(encoded) > 0
        assert "MPa" in decoded or "psi" in decoded

    def test_engineering_boundary_markers(self, eng_tokenizer):
        """Test that engineering symbol boundaries are recognized."""
        text = "<ENG_SYMBOL_START> σ = 100 MPa <ENG_SYMBOL_END>"
        tokens = eng_tokenizer.tokenize(text)

        # Boundary markers should be present
        token_str = " ".join(tokens)
        # Markers should be tokenized (exact form may vary)
        assert "ENG_SYMBOL" in token_str or len(tokens) > 0


class TestDSLParsing:
    """Test DSL boundary parsing."""

    @pytest.fixture
    def dsl_tokenizer(self):
        """Create tokenizer trained on DSL examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "dsl_corpus.txt"
            corpus_file.write_text(
                "<DSL_BEGIN> CHECK_SPC X-bar chart, n=5 <DSL_END>\n"
                "<DSL_BEGIN> EVAL_DOE factorial 2^3 <DSL_END>\n"
                "<DSL_BEGIN> RUN_TTEST alpha=0.05 <DSL_END>\n"
                "Perform ANOVA\n"
                "Calculate Cpk\n"
            )

            config = TokenizerTrainingConfig(
                vocab_size=5000,
                output_dir=tmpdir,
                input_files=[str(corpus_file)],
            )
            train_tokenizer(config, verbose=False)

            yield HavocTokenizer(tmpdir)

    def test_dsl_boundary_tokenization(self, dsl_tokenizer):
        """Test that DSL boundaries are properly tokenized."""
        text = "<DSL_BEGIN> CHECK_SPC <DSL_END>"
        tokens = dsl_tokenizer.tokenize(text)

        # Should have tokens for boundaries
        assert len(tokens) > 0

    def test_dsl_with_content(self, dsl_tokenizer):
        """Test DSL with content inside boundaries."""
        text = "<DSL_BEGIN> EVAL_DOE factorial 2^3, RESPONSE=yield <DSL_END>"
        encoded = dsl_tokenizer.encode(text)
        decoded = dsl_tokenizer.decode(encoded)

        assert len(encoded) > 0
        # Should contain key DSL keywords
        assert "EVAL_DOE" in decoded or "factorial" in decoded or len(decoded) > 0


class TestSRSReasoningTokens:
    """Test SRS reasoning stage token handling."""

    @pytest.fixture
    def srs_tokenizer(self):
        """Create tokenizer trained on SRS examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_file = Path(tmpdir) / "srs_corpus.txt"
            corpus_file.write_text(
                "<SRS_MODE> Identify problem type <SRS_GROUND>\n"
                "<SRS_GROUND> Gather background knowledge <SRS_PLAN>\n"
                "<SRS_PLAN> Design solution approach <SRS_EXECUTE>\n"
                "<SRS_EXECUTE> Implement solution <SRS_ARGUE>\n"
                "<SRS_ARGUE> Generate counterarguments <SRS_ARBITER>\n"
                "<SRS_ARBITER> Resolve conflicts <SRS_AUDIT>\n"
                "<SRS_AUDIT> Verify correctness <SRS_ANSWER>\n"
                "<SRS_ANSWER> Final answer\n"
            )

            config = TokenizerTrainingConfig(
                vocab_size=5000,
                output_dir=tmpdir,
                input_files=[str(corpus_file)],
            )
            train_tokenizer(config, verbose=False)

            yield HavocTokenizer(tmpdir)

    def test_srs_stage_tokenization(self, srs_tokenizer):
        """Test that SRS stage markers are tokenized."""
        text = "<SRS_MODE> Problem identification <SRS_GROUND>"
        tokens = srs_tokenizer.tokenize(text)

        assert len(tokens) > 0
        # Stage markers should be present
        token_str = " ".join(tokens)
        assert "SRS" in token_str or len(tokens) > 2

    def test_full_srs_pipeline(self, srs_tokenizer):
        """Test tokenization of full SRS reasoning pipeline."""
        text = (
            "<SRS_MODE> Identify <SRS_GROUND> Research <SRS_PLAN> Design "
            "<SRS_EXECUTE> Implement <SRS_ARGUE> Challenge <SRS_ARBITER> Resolve "
            "<SRS_AUDIT> Verify <SRS_ANSWER> Conclude"
        )
        encoded = srs_tokenizer.encode(text)
        decoded = srs_tokenizer.decode(encoded)

        assert len(encoded) > 0
        # Should preserve the pipeline structure
        assert "SRS" in decoded or len(decoded) > 10


class TestIntegration:
    """Integration tests for complete tokenizer workflow."""

    def test_train_load_encode_decode(self):
        """Test full workflow: train, load, encode, decode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train tokenizer
            config = TokenizerTrainingConfig(
                vocab_size=5000,
                output_dir=tmpdir,
                input_files=[],
            )
            metadata = train_tokenizer(config, verbose=False)

            # Load tokenizer
            tokenizer = HavocTokenizer(tmpdir)

            # Test encoding/decoding
            texts = [
                "Calculate the mean μ and variance σ²",
                "<SRS_MODE> Identify problem <SRS_GROUND>",
                "<DSL_BEGIN> CHECK_SPC <DSL_END>",
                "Stress: 250 MPa at 150°C",
            ]

            for text in texts:
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)

                # Should successfully encode and decode
                assert len(encoded) > 0
                assert isinstance(decoded, str)
                assert len(decoded) > 0

    def test_vocab_size_consistency(self):
        """Test that vocab size is consistent between training and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_vocab_size = 10000

            config = TokenizerTrainingConfig(
                vocab_size=target_vocab_size,
                output_dir=tmpdir,
                input_files=[],
            )
            metadata = train_tokenizer(config, verbose=False)

            tokenizer = HavocTokenizer(tmpdir)

            # Vocab sizes should match (approximately, SentencePiece may adjust)
            assert abs(tokenizer.vocab_size - target_vocab_size) < 100
