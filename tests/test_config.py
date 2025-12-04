from havoc_core.config import HavocConfig, TokenizerTrainingConfig


def test_havoc_config_defaults():
    cfg = HavocConfig.havoc_2b()
    assert cfg.d_model == 2560
    assert cfg.num_layers == 20
    assert cfg.max_seq_len == 1024
    assert cfg.attention.num_heads == 32
    assert cfg.attention.num_kv_heads == 4
    assert cfg.mlp.hidden_dim == 10240


def test_tokenizer_special_tokens():
    tok_cfg = TokenizerTrainingConfig()
    assert "<pad>" in tok_cfg.special_tokens
    assert tok_cfg.vocab_size > 0

