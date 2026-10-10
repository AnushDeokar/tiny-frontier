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
from tiny_frontier.model.norms import RMSNorm  # noqa: E402


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
