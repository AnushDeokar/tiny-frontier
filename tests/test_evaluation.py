import pytest
import torch

from conftest import tiny_model
from tiny_frontier.config import Config
from tiny_frontier.evaluation import loss_by_position, max_eval_batches
from tiny_frontier.training.train import evaluate, parse_args


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_loss_by_position_averages_to_the_val_loss(device):
    # Same windows, same model: the mean over positions must equal the plain mean loss.
    # Also runs on the Mac GPU (MPS) when present: a float64 bug only showed up there.
    model = tiny_model().to(device)
    tokens = torch.randint(0, 64, (5_000,)).numpy().astype("uint16")
    cfg = Config()
    cfg.model.max_seq_len, cfg.train.batch_size, cfg.train.eval_batches = 32, 4, 6
    by_pos = loss_by_position(model, tokens, 4, 32, 6, device)
    assert by_pos.shape == (32,)
    assert abs(by_pos.mean().item() - evaluate(model, tokens, cfg, device)) < 1e-4


def test_window_only_changes_loss_beyond_the_window():
    """The core idea behind the loss-by-position chart: before position W a
    local model sees exactly what a global one sees, so with identical weights
    the losses are identical there and may differ only from W on."""
    W = 8
    torch.manual_seed(0)
    global_model = tiny_model()
    torch.manual_seed(0)
    local_model = tiny_model(layer_pattern="L", window=W)
    tokens = torch.randint(0, 64, (5_000,)).numpy().astype("uint16")
    a = loss_by_position(global_model, tokens, 4, 32, 6, "cpu")
    b = loss_by_position(local_model, tokens, 4, 32, 6, "cpu")
    torch.testing.assert_close(a[:W], b[:W])
    assert not torch.allclose(a[W:], b[W:])


def test_max_eval_batches():
    assert max_eval_batches(range(8193), batch_size=32, block_size=256) == 1
    assert max_eval_batches(range(8192), batch_size=32, block_size=256) == 0


def test_pattern_and_window_options():
    args = parse_args(["--config", "configs/control.yaml", "--pattern", "LLGLLG", "--window", "32"])
    assert args.pattern == "LLGLLG" and args.window == 32
