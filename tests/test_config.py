from havoc_core.config import HavocConfig, TokenizerTrainingConfig


def test_havoc_config_defaults():
    cfg = HavocConfig.havoc_7b()
    assert cfg.d_model == 4096
    assert cfg.attention.num_heads == 32
    assert cfg.mlp.hidden_dim == 11008


def test_tokenizer_special_tokens():
    tok_cfg = TokenizerTrainingConfig()
    assert "<pad>" in tok_cfg.special_tokens
    assert tok_cfg.vocab_size > 0
