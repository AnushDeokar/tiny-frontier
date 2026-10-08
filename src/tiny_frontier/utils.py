def kv_cache_bytes(model_cfg, seq_len, bytes_per_value=4):
    """Memory the KV cache needs to hold `seq_len` tokens (all layers global).

    Per layer we store K and V (the factor 2), each [seq_len, n_kv_heads, head_dim].
    bytes_per_value: 4 for fp32, 2 for fp16/bf16.
    Experiment 1 extends this: local layers cache min(seq_len, window) tokens.
    """
    n_kv_heads = model_cfg.n_kv_heads or model_cfg.n_heads
    head_dim = model_cfg.head_dim or model_cfg.d_model // model_cfg.n_heads
    return 2 * model_cfg.n_layers * seq_len * n_kv_heads * head_dim * bytes_per_value
