import pytest
import torch

from conftest import tiny_model
from tiny_frontier.model.attention import CausalSelfAttention, KVCache
from tiny_frontier.model.rope import RotaryEmbedding


def test_causality():
    """Changing a future token must not change the logits of earlier positions.

    If this fails, the model can "see the answer" during training and the
    loss will look great while the model learns nothing useful.
    """
    model = tiny_model().eval()
    x = torch.randint(0, 64, (1, 16))
    t = 7
    x_changed = x.clone()
    x_changed[0, t + 1 :] = (x_changed[0, t + 1 :] + 1) % 64

    with torch.no_grad():
        a = model(x)
        b = model(x_changed)

    torch.testing.assert_close(a[:, : t + 1], b[:, : t + 1])
    # Sanity check that the change did reach later positions.
    assert not torch.allclose(a[:, t + 1], b[:, t + 1])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"n_kv_heads": 2}, {"n_kv_heads": 1}, {"head_dim": 16}],
    ids=["mha", "gqa-2", "mqa-1", "head_dim-16"],
)
def test_cached_decoding_equals_full_forward(kwargs):
    """Generating token by token with the KV cache must give the same logits
    as running the whole sequence at once. This catches RoPE offset bugs,
    mask bugs and GQA head-expansion bugs."""
    model = tiny_model(**kwargs).eval()
    n_layers = len(model.blocks)
    x = torch.randint(0, 64, (2, 20))

    with torch.no_grad():
        full = model(x)

        cache = KVCache(n_layers)
        prefill = model(x[:, :5], kv_cache=cache)  # prompt in one go
        steps = [model(x[:, t : t + 1], kv_cache=cache) for t in range(5, 20)]
        cached = torch.cat([prefill] + steps, dim=1)

    torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)


def test_kv_cache_stores_only_kv_heads():
    # The memory saving of GQA: the cache holds n_kv_heads, not n_heads.
    model = tiny_model(n_kv_heads=2).eval()
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 6)), kv_cache=cache)
    assert cache.k[0].shape == (1, 2, 6, 8)  # [B, n_kv_heads, T, head_dim]


def test_gqa_equals_mha_with_shared_kv_weights():
    """GQA is just MHA where groups of query heads share one K/V head.

    Build an MHA layer whose K/V weights are copies of the GQA layer's
    (each K/V head duplicated n_rep times). Outputs must match exactly.
    """
    d_model, n_heads, n_kv, head_dim = 32, 4, 2, 8
    rope = RotaryEmbedding(head_dim, 64)
    gqa = CausalSelfAttention(d_model, n_heads, rope, n_kv_heads=n_kv)
    mha = CausalSelfAttention(d_model, n_heads, rope)

    def expand(w):
        # [n_kv * head_dim, d_model] -> [n_heads * head_dim, d_model]
        w = w.view(n_kv, head_dim, d_model)
        return w.repeat_interleave(n_heads // n_kv, dim=0).reshape(-1, d_model)

    with torch.no_grad():
        mha.Wq.weight.copy_(gqa.Wq.weight)
        mha.Wo.weight.copy_(gqa.Wo.weight)
        mha.Wk.weight.copy_(expand(gqa.Wk.weight))
        mha.Wv.weight.copy_(expand(gqa.Wv.weight))

        x = torch.randn(2, 10, d_model)
        torch.testing.assert_close(gqa(x), mha(x))
