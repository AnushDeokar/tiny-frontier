from tiny_frontier.model.model import expand_layer_pattern


def kv_cache_bytes(model_cfg, seq_len, bytes_per_value=4):
    """Memory the KV cache needs to hold a `seq_len`-token context.

    Per layer we store K and V (the factor 2), each [tokens, n_kv_heads, head_dim].
    A global layer must keep all seq_len tokens; a local layer only ever looks
    at the last `window` tokens, so it needs to keep at most that many.
    bytes_per_value: 4 for fp32, 2 for fp16/bf16.
    """
    n_kv_heads = model_cfg.n_kv_heads or model_cfg.n_heads
    head_dim = model_cfg.head_dim or model_cfg.d_model // model_cfg.n_heads
    layer_types = expand_layer_pattern(model_cfg.layer_pattern, model_cfg.n_layers)
    cached_tokens = sum(
        min(seq_len, model_cfg.window) if t == "local" else seq_len for t in layer_types
    )
    return 2 * cached_tokens * n_kv_heads * head_dim * bytes_per_value
