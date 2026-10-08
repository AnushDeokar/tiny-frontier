import math

import torch

from conftest import tiny_model
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.training.loss import lm_loss


def control_model():
    # The plan's control: 6 layers, width 192, 6 heads, SwiGLU hidden 512.
    return LlamaModel(
        vocab_size=4096, d_model=192, n_heads=6, d_ff=512, n_layers=6, max_seq_len=256
    )


def test_control_param_count():
    # Pinning the exact count means any accidental architecture change
    # (a stray bias, an untied head) fails loudly instead of silently.
    model = control_model()
    assert model.num_params() == 3_443_136
    assert model.num_params(non_embedding=True) == 2_656_704


def test_embeddings_are_tied():
    model = control_model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_initial_loss_is_near_uniform():
    # A fresh model should be "maximally unsure": every token about equally
    # likely, so the loss is ln(vocab_size) = ln(4096) ~ 8.32.
    # Much higher means a bad init (e.g. N(0, 1) embeddings).
    model = control_model().eval()
    x = torch.randint(0, 4096, (4, 64))
    with torch.no_grad():
        loss = lm_loss(model(x[:, :-1]), x[:, 1:])
    assert abs(loss.item() - math.log(4096)) < 0.2


def test_forward_beyond_training_context():
    # Trained on 32 tokens here, but the RoPE table is longer, so evaluating
    model = tiny_model(max_seq_len=32).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 64, (1, 300)))
    assert out.shape == (1, 300, 64)


def test_overfit_one_batch():
    """The classic smoke test: a model must be able to memorise one tiny batch.

    If loss won't go near zero here, something is broken in the forward pass,
    the loss, or the gradient flow, and no amount of real training will help.
    """
    model = tiny_model()
    x = torch.randint(0, 64, (2, 17))
    inputs, targets = x[:, :-1], x[:, 1:]  # predict the next token
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(300):
        loss = lm_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"could not overfit one batch, final loss {loss.item():.3f}"
