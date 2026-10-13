"""Rolling KV cache: local layers keep only their last W keys/values.

Dropping old entries must not change a single output (a local layer can't
see them anyway), and must make the cache stop growing.
"""

from dataclasses import asdict

import torch

from tiny_frontier.config import ModelConfig
from tiny_frontier.generate import generate
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.utils import kv_cache_bytes

CFG = ModelConfig(
    vocab_size=64, d_model=32, n_heads=4, d_ff=64, n_layers=3, max_seq_len=32,
    layer_pattern="LLG", window=4,
)


def make_model():
    return LlamaModel(**asdict(CFG)).eval()


@torch.no_grad()
def test_rolling_cache_gives_exactly_the_full_forward():
    # 40 tokens with a window of 4: the local layers' caches roll over many times.
    model = make_model()
    x = torch.randint(0, 64, (2, 40))
    full = model(x)

    cache = model.new_cache(rolling=True)
    # Prompt of 12 tokens in one go (longer than the window), then 1 at a time.
    steps = [model(x[:, :12], kv_cache=cache)] + [model(x[:, t : t + 1], kv_cache=cache) for t in range(12, 40)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_rolling_cache_stops_growing_in_local_layers_only():
    model = make_model()
    cache = model.new_cache(rolling=True)
    model(torch.randint(0, 64, (1, 50)), kv_cache=cache)
    for i, kind in enumerate(model.layer_types):
        assert cache.length(i) == 50  # every layer has *processed* 50 tokens
        assert cache.stored(i) == (4 if kind == "local" else 50)  # but local layers keep 4


@torch.no_grad()
def test_measured_cache_memory_matches_the_formula():
    # The cache's real bytes after 50 tokens equal kv_cache_bytes() (batch of 1):
    # 2 local layers x 4 tokens + 1 global layer x 50 tokens, K and V, fp32.
    model = make_model()
    cache = model.new_cache(rolling=True)
    model(torch.randint(0, 64, (1, 50)), kv_cache=cache)
    assert cache.nbytes() == kv_cache_bytes(CFG, 50)

    keep_all = model.new_cache(rolling=False)
    model(torch.randint(0, 64, (1, 50)), kv_cache=keep_all)
    assert keep_all.nbytes() == kv_cache_bytes(ModelConfig(**{**asdict(CFG), "layer_pattern": None, "window": None}), 50)


@torch.no_grad()
def test_generation_same_with_and_without_rolling():
    model = make_model()
    prompt = torch.randint(0, 64, (1, 6))
    rolling = generate(model, prompt, 30, temperature=0, kv_cache=model.new_cache(rolling=True))
    keep_all = generate(model, prompt, 30, temperature=0, kv_cache=model.new_cache(rolling=False))
    no_cache = generate(model, prompt, 30, temperature=0, use_cache=False)
    assert torch.equal(rolling, keep_all) and torch.equal(rolling, no_cache)
