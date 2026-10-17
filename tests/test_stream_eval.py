import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import tiny_model
from tiny_frontier.evaluation import first_token_attention, make_streams, stream_losses
from tiny_frontier.model.attention import KVCache, StreamCache


def test_stream_losses_with_full_cache_equal_normal_evaluation():
    # Feeding one token at a time with a full cache must give exactly the
    # per-token losses of one ordinary forward pass (teacher forcing in both).
    model = tiny_model().eval()
    x = torch.randint(0, 64, (3, 25))
    got = stream_losses(model, x, KVCache(2), "cpu")
    with torch.no_grad():
        logits = model(x)[:, :-1]
    want = F.cross_entropy(logits.reshape(-1, 64), x[:, 1:].reshape(-1), reduction="none").view(3, 24)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_stream_losses_shape_with_a_sink_cache():
    model = tiny_model().eval()
    x = torch.randint(0, 64, (2, 50))
    got = stream_losses(model, x, StreamCache(2, n_sink=4, window=8), "cpu")
    assert got.shape == (2, 49) and torch.isfinite(got).all()


def test_make_streams_are_consecutive_slices():
    tokens = np.arange(1000, dtype=np.uint16)
    s = make_streams(tokens, n_streams=3, length=100)
    assert s.shape == (3, 100) and s.dtype == torch.int64
    assert s[1, 0].item() == 100 and s[2, -1].item() == 299


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_first_token_attention(device):
    # Also on the Mac GPU: the same float64 bug hit this function on MPS.
    model = tiny_model().to(device)
    tokens = np.random.randint(0, 64, 5000).astype(np.uint16)
    mass, sink = first_token_attention(model, tokens, 4, 32, 2, device, min_query=4)
    assert mass.shape == (2, 4) and sink is None
    assert ((mass >= 0) & (mass <= 1)).all()
    assert all(not b.attn.capture and b.attn.last_weights is None for b in model.blocks)  # cleaned up


def test_learned_sink_share_is_measured():
    # A huge sink logit: every head sends ~all its attention to the sink,
    # so ~nothing lands on token 0.
    model = tiny_model(learned_sink=True)
    with torch.no_grad():
        for b in model.blocks:
            b.attn.sink_logit.fill_(30.0)
    tokens = np.random.randint(0, 64, 5000).astype(np.uint16)
    mass, sink = first_token_attention(model, tokens, 4, 32, 2, "cpu", min_query=4)
    assert (sink > 0.999).all() and (mass < 1e-3).all()
