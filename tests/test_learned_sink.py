"""gpt-oss style learned attention sink: one extra softmax "column" per head
with a learnable score and no value vector."""

import torch

from conftest import tiny_model
from tiny_frontier.model.attention import CausalSelfAttention
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.model.rope import RotaryEmbedding
from tiny_frontier.training.loss import lm_loss


@torch.no_grad()
def test_sink_at_minus_infinity_is_normal_attention():
    # exp(-inf) = 0: the sink column gets no weight, so softmax is unchanged.
    # The sink logits are created as zeros (no random numbers), so with the
    # same seed both models get identical weights everywhere else.
    torch.manual_seed(0)
    plain = tiny_model().eval()
    torch.manual_seed(0)
    sink = tiny_model(learned_sink=True).eval()
    for block in sink.blocks:
        block.attn.sink_logit.fill_(float("-inf"))
    x = torch.randint(0, 64, (2, 16))
    torch.testing.assert_close(sink(x), plain(x))


@torch.no_grad()
def test_large_sink_means_attend_to_nothing():
    # A huge sink score takes ~all of the softmax, the real tokens get ~0,
    # and the head's output is ~0: the "none of the above" option normal
    # softmax doesn't have.
    attn = CausalSelfAttention(32, 4, RotaryEmbedding(8, 64), learned_sink=True)
    attn.sink_logit.fill_(30.0)
    out = attn(torch.randn(2, 10, 32))
    assert out.abs().max() < 1e-6


def test_sink_adds_one_parameter_per_head_per_layer():
    base = dict(vocab_size=4096, d_model=192, n_heads=6, d_ff=512, n_layers=6, max_seq_len=256)
    assert LlamaModel(**base, learned_sink=True).num_params() == 3_443_136 + 6 * 6


def test_sink_logit_learns():
    model = tiny_model(learned_sink=True)
    x = torch.randint(0, 64, (2, 17))
    lm_loss(model(x[:, :-1]), x[:, 1:]).backward()
    assert all(b.attn.sink_logit.grad is not None and b.attn.sink_logit.grad.abs().sum() > 0 for b in model.blocks)


@torch.no_grad()
def test_cached_decoding_with_learned_sink():
    model = tiny_model(learned_sink=True, n_layers=3, layer_pattern="LLG", window=4).eval()
    for block in model.blocks:
        block.attn.sink_logit.normal_()  # non-zero, different per head
    x = torch.randint(0, 64, (2, 20))
    full = model(x)
    cache = model.new_cache()
    steps = [model(x[:, :6], kv_cache=cache)] + [model(x[:, t : t + 1], kv_cache=cache) for t in range(6, 20)]
    torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-5, rtol=1e-5)
