"""Streaming cache policies (Experiment 2): full, rolling(W), sink(k, W),
with keys stored before RoPE and positions assigned by cache slot."""

import pytest
import torch

from conftest import tiny_model
from tiny_frontier.model.attention import StreamCache


@torch.no_grad()
def stream(model, x, cache):
    """Feed x one token at a time, like the stream evaluator."""
    return torch.cat([model(x[:, t : t + 1], kv_cache=cache) for t in range(x.shape[1])], dim=1)


@torch.no_grad()
def test_full_policy_equals_the_normal_forward():
    # Nothing evicted: slot positions are the text positions.
    model = tiny_model().eval()
    x = torch.randint(0, 64, (2, 30))
    torch.testing.assert_close(stream(model, x, StreamCache(2)), model(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("length", [12, 20])
@torch.no_grad()
def test_sink_policy_on_a_short_stream_equals_full(length):
    # The plan's test: until sinks + window are full, nothing is evicted, so
    # sink(4, 16) must give exactly the full-cache answer.
    model = tiny_model().eval()
    x = torch.randint(0, 64, (2, length))
    sink = stream(model, x, StreamCache(2, n_sink=4, window=16))
    torch.testing.assert_close(sink, model(x), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_sink_policy_keeps_the_first_tokens_forever():
    model = tiny_model().eval()
    x = torch.randint(0, 64, (1, 60))
    cache = StreamCache(2, n_sink=4, window=16)
    stream(model, x[:, :4], cache)
    first_keys = cache.k[0][..., :4, :].clone()
    stream(model, x[:, 4:], cache)
    assert cache.stored(0) == 4 + 16  # memory stays bounded
    assert torch.equal(cache.k[0][..., :4, :], first_keys)  # the sink tokens never left


@torch.no_grad()
def test_rolling_policy_is_exact_for_a_model_trained_with_that_window():
    # A local model with window 8 never looks further back than 8 tokens, and
    # RoPE only cares about distances, so rolling(8) with slot positions must
    # reproduce its full forward pass exactly, even after 40 tokens.
    model = tiny_model(layer_pattern="L", window=8).eval()
    x = torch.randint(0, 64, (2, 40))
    torch.testing.assert_close(stream(model, x, StreamCache(2, window=8)), model(x), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_rolling_policy_changes_a_global_model():
    # For a model trained with full attention, evicting tokens does change
    # the output: this is the situation Experiment 2 measures.
    model = tiny_model().eval()
    x = torch.randint(0, 64, (1, 40))
    rolled = stream(model, x, StreamCache(2, window=8))
    assert not torch.allclose(rolled[:, 20:], model(x)[:, 20:], atol=1e-4)


@torch.no_grad()
def test_stream_cache_with_learned_sink_model():
    model = tiny_model(learned_sink=True).eval()
    out = stream(model, torch.randint(0, 64, (2, 40)), StreamCache(2, window=8))
    assert torch.isfinite(out).all()


def test_sink_cache_needs_a_window():
    with pytest.raises(ValueError):
        StreamCache(2, n_sink=4)
