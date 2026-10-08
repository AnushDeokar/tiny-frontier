import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_seed():
    # Every test starts from the same random state, so a failure is reproducible.
    torch.manual_seed(0)


def tiny_model(**kwargs):
    """A model small enough that every test runs in well under a second on CPU."""
    from tiny_frontier.model.model import LlamaModel

    config = dict(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        max_seq_len=32,
    )
    config.update(kwargs)
    return LlamaModel(**config)
