"""Gemma 3 building blocks, checked against Hugging Face's own implementation.

These use random weights, so they need the `transformers` code but not the
downloaded Gemma checkpoint. If the architecture matches on random weights,
the real weights are only the final confirmation.
"""

import math

import pytest
import torch

hf = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")

from conftest import tiny_model  # noqa: E402
from tiny_frontier.model.attention import CausalSelfAttention, KVCache  # noqa: E402
from tiny_frontier.model.block import DecoderBlock  # noqa: E402
from tiny_frontier.model.model import LlamaModel  # noqa: E402
from tiny_frontier.model.norms import RMSNorm  # noqa: E402
from tiny_frontier.model.rope import RotaryEmbedding  # noqa: E402


def test_offset_rmsnorm_matches_gemma():
    ours = RMSNorm(64, eps=1e-6, offset=True)
    ref = hf.Gemma3RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        w = torch.randn(64) * 0.1  # trained Gemma norm weights are small numbers around 0
        ours.weight.copy_(w)
        ref.weight.copy_(w)
    x = torch.randn(2, 7, 64)
    torch.testing.assert_close(ours(x), ref(x))


def test_offset_and_plain_rmsnorm_start_identical():
    # Llama stores w = 1, Gemma stores w = 0 and adds 1. A fresh norm is the
    # same function either way; only the stored number differs.
    x = torch.randn(3, 64)
    torch.testing.assert_close(RMSNorm(64, offset=True)(x), RMSNorm(64, offset=False)(x))
    assert torch.all(RMSNorm(64, offset=True).weight == 0)


def test_norm_convention_does_not_change_a_fresh_model():
    # Norm weights are created with ones/zeros (no random numbers), so with
    # the same seed both models get identical random weights everywhere else.
    torch.manual_seed(0)
    plain = tiny_model().eval()
    torch.manual_seed(0)
    gemma_style = tiny_model(norm_offset=True).eval()
    x = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(plain(x), gemma_style(x))


def test_embed_scale_matches_gemma():
    model = tiny_model(embed_scale=True)
    d_model = model.token_embedding.embedding_dim
    ref = hf.Gemma3TextScaledWordEmbedding(64, d_model, padding_idx=0, embed_scale=d_model**0.5)
    with torch.no_grad():
        ref.weight.copy_(model.token_embedding.weight)
    ids = torch.randint(0, 64, (2, 9))
    torch.testing.assert_close(model.embed(ids), ref(ids))
    assert model.embed_multiplier == pytest.approx(math.sqrt(d_model))


def test_flags_off_keep_the_control_unchanged():
    # The defaults must reproduce the control exactly (see also the exact
    # parameter-count test in test_model.py).
    model = tiny_model()
    assert model.embed_multiplier == 1.0
    assert all(not m.offset for m in model.modules() if isinstance(m, RMSNorm))
    for block in model.blocks:
        assert not block.post_norm and not hasattr(block, "post_attn_norm")
        assert not block.attn.qk_norm and not hasattr(block.attn, "q_norm")


def test_qk_norm_makes_attention_ignore_the_size_of_q_and_k():
    """With QK-norm, q and k are rescaled to a fixed size before q.k, so
    making Wq or Wk bigger can't make the softmax sharper. Without it, the
    same change makes the attention pattern far more peaked."""
    x = torch.randn(2, 10, 32)
    for qk_norm in (True, False):
        torch.manual_seed(0)
        attn = CausalSelfAttention(32, 4, RotaryEmbedding(8, 64), qk_norm=qk_norm)
        with torch.no_grad():
            before = attn(x)
            attn.Wq.weight.mul_(5.0)
            attn.Wk.weight.mul_(3.0)
            after = attn(x)
        unchanged = torch.allclose(before, after, atol=1e-5)
        assert unchanged == qk_norm, f"qk_norm={qk_norm}"


def test_post_norm_block_order():
    # Gemma 3 decoder layer, in order:
    #   x = x + post_attn_norm(attn(norm_1(x)))
    #   x = x + post_ff_norm(ff(norm_2(x)))
    block = DecoderBlock(32, 4, 64, RotaryEmbedding(8, 64), post_norm=True, norm_offset=True)
    with torch.no_grad():
        for m in block.modules():
            if isinstance(m, RMSNorm):
                m.weight.normal_(0, 0.1)  # non-trivial scales, so a skipped norm would show
        x = torch.randn(2, 6, 32)
        h = x + block.post_attn_norm(block.attn(block.norm_1(x)))
        expected = h + block.post_ff_norm(block.ff(block.norm_2(h)))
        torch.testing.assert_close(block(x), expected)


def test_cached_decoding_with_gemma_norms():
    model = tiny_model(n_kv_heads=1, qk_norm=True, post_norm=True, norm_offset=True, embed_scale=True).eval()
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        full = model(x)
        cache = KVCache(len(model.blocks))
        steps = [model(x[:, :4], kv_cache=cache)] + [model(x[:, t : t + 1], kv_cache=cache) for t in range(4, 16)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-5)


def test_gemma_270m_shape_has_exactly_its_parameter_count():
    """Build a model with Gemma 3 270M's real shape and Gemma flags.

    The count must equal Google's 268,098,176 (from the checkpoint metadata).
    That confirms every weight matrix and norm exists with the right size:
    tied embeddings, no biases, 4 norms per block, QK-norm of size head_dim.
    'meta' device: shapes only, no memory, so this is instant.
    """
    with torch.device("meta"):
        model = LlamaModel(
            vocab_size=262_144,
            d_model=640,
            n_heads=4,
            n_kv_heads=1,
            head_dim=256,
            d_ff=2048,
            n_layers=18,
            max_seq_len=512,
            norm_offset=True,
            embed_scale=True,
            qk_norm=True,
            post_norm=True,
        )
    assert model.num_params() == 268_098_176
