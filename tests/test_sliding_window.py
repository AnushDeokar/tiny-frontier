"""Sliding-window (local) attention, the layer pattern, and per-layer RoPE bases.

Used by Gemma 3 (5 local : 1 global, window 512) and by Experiment 1.
"""

import pytest
import torch

from conftest import tiny_model
from tiny_frontier.config import ModelConfig
from tiny_frontier.model.attention import CausalSelfAttention, KVCache
from tiny_frontier.model.model import expand_layer_pattern
from tiny_frontier.model.rope import RotaryEmbedding
from tiny_frontier.utils import kv_cache_bytes


@torch.no_grad()
def test_window_sees_exactly_the_last_w_tokens():
    """Nudge each input position in turn and see which ones reach query i.

    With W = 4 and i = 10, exactly positions 7, 8, 9, 10 may matter.
    """
    W, i = 4, 10
    torch.manual_seed(0)
    attn = CausalSelfAttention(32, 4, RotaryEmbedding(8, 64), window=W).eval()
    x = torch.randn(1, 16, 32)
    base = attn(x)[0, i]
    for j in range(16):
        x2 = x.clone()
        x2[0, j] += 1.0
        changed = not torch.allclose(attn(x2)[0, i], base, atol=1e-6)
        assert changed == (i - W + 1 <= j <= i), f"position {j}"


@torch.no_grad()
def test_window_as_long_as_the_context_is_plain_attention():
    # A window that can't cut anything off must give exactly the control's output.
    torch.manual_seed(0)
    control = tiny_model().eval()
    torch.manual_seed(0)
    local = tiny_model(layer_pattern="L", window=32).eval()
    x = torch.randint(0, 64, (2, 32))
    torch.testing.assert_close(local(x), control(x))


@torch.no_grad()
def test_receptive_field_grows_with_depth():
    """Each local layer looks back W-1 tokens, and stacking layers adds that up:
    after 2 layers with W = 4, position i can be influenced by i-6 but not i-7.
    This is why local layers still "see" far back in a deep model, indirectly."""
    W, i = 4, 12
    model = tiny_model(n_layers=2, layer_pattern="L", window=W).eval()
    x = torch.randint(0, 64, (1, 16))
    base = model(x)[0, i]

    def reaches(j):
        x2 = x.clone()
        x2[0, j] = (x2[0, j] + 1) % 64
        return not torch.allclose(model(x2)[0, i], base, atol=1e-6)

    assert reaches(i - 2 * (W - 1))
    assert not reaches(i - 2 * (W - 1) - 1)


@torch.no_grad()
def test_cached_decoding_with_local_and_global_layers():
    # The window must be measured in absolute positions, also when the
    # queries are only the newest tokens and the keys come from the cache.
    model = tiny_model(
        n_layers=3, layer_pattern="LLG", window=4, rope_base=1e6, rope_base_local=1e4
    ).eval()
    x = torch.randint(0, 64, (2, 20))
    full = model(x)
    cache = KVCache(3)
    steps = [model(x[:, :5], kv_cache=cache)] + [model(x[:, t : t + 1], kv_cache=cache) for t in range(5, 20)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-5)


def test_layer_pattern_matches_gemma_3_270m():
    # Copied from the real google/gemma-3-270m config.json "layer_types".
    hf = ["sliding_attention"] * 5 + ["full_attention"]
    hf = hf * 3
    ours = expand_layer_pattern("LLLLLG", 18)
    assert ours == ["local" if t == "sliding_attention" else "global" for t in hf]


@pytest.mark.parametrize("pattern,n", [("LLX", 6), ("LLLLL", 18), ("", 6)])
def test_bad_layer_patterns_are_rejected(pattern, n):
    with pytest.raises(ValueError):
        expand_layer_pattern(pattern, n)


def test_local_layers_need_a_window():
    with pytest.raises(ValueError, match="window"):
        tiny_model(layer_pattern="L")


def test_each_layer_gets_its_window_and_rope_base():
    model = tiny_model(n_layers=6, layer_pattern="LLLLLG", window=8, rope_base=1e6, rope_base_local=1e4)
    for block, kind in zip(model.blocks, model.layer_types):
        if kind == "local":
            assert block.attn.window == 8 and block.attn.rope.base == 1e4
        else:
            assert block.attn.window is None and block.attn.rope.base == 1e6


def test_kv_cache_bytes():
    # Control: all global, unchanged from the numbers in every summary.json.
    assert kv_cache_bytes(ModelConfig(), 256) == 2_359_296
    assert kv_cache_bytes(ModelConfig(), 4096) == 37_748_736

    # Gemma 3 270M at a 32k context, fp32: 3 global layers keep 32,768 tokens,
    # 15 local layers keep only 512 each.
    gemma = ModelConfig(
        vocab_size=262_144, d_model=640, n_heads=4, n_kv_heads=1, head_dim=256,
        d_ff=2048, n_layers=18, layer_pattern="LLLLLG", window=512,
    )
    all_global = ModelConfig(**{**gemma.__dict__, "layer_pattern": None, "window": None})
    assert kv_cache_bytes(gemma, 32_768) == 2 * (3 * 32_768 + 15 * 512) * 1 * 256 * 4  # ~207 MB
    assert kv_cache_bytes(all_global, 32_768) == 2 * 18 * 32_768 * 1 * 256 * 4  # ~1.15 GB
