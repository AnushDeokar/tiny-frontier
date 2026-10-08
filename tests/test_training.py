import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_frontier.config import TrainConfig, load_config
from tiny_frontier.data.loader import get_batch, iter_eval_batches
from tiny_frontier.training.optim import build_optimizer, lr_at
from conftest import tiny_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100, min_lr_ratio=0.1)
    total = 1000
    assert lr_at(0, total, cfg) == pytest.approx(1e-5)  # warmup starts tiny
    assert lr_at(99, total, cfg) == pytest.approx(1e-3)  # peak at end of warmup
    assert lr_at(550, total, cfg) == pytest.approx(0.55e-3)  # halfway down the cosine
    assert lr_at(999, total, cfg) == pytest.approx(1e-4, rel=1e-2)  # ends at 10% of peak


def test_weight_decay_groups():
    model = tiny_model()
    opt = build_optimizer(model, TrainConfig())
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    # The tied embedding must appear exactly once across both groups.
    n_in_groups = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_in_groups == model.num_params()


def test_control_config_loads():
    cfg = load_config(CONFIGS / "control.yaml")
    assert cfg.model.d_model == 192
    assert isinstance(cfg.train.lr, float)
    assert cfg.tokens_per_step == 8192
    assert cfg.total_steps == 50_000_000 // 8192


def test_config_rejects_typos(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  learning_rate: 0.001\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(path)


def test_eval_batches_are_fixed():
    tokens = np.arange(10_000, dtype=np.uint16)
    a = list(iter_eval_batches(tokens, batch_size=4, block_size=16, n_batches=3))
    b = list(iter_eval_batches(tokens, batch_size=4, block_size=16, n_batches=3))
    for (xa, ya), (xb, yb) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    x, y = a[0]
    assert torch.equal(y[:, :-1], x[:, 1:])  # y is x shifted by one
    assert x.dtype == torch.int64


def test_train_batches_depend_only_on_seed():
    tokens = np.arange(10_000, dtype=np.uint16)
    x1, _ = get_batch(tokens, 4, 16, generator=torch.Generator().manual_seed(1))
    torch.rand(100)  # unrelated random draw must not change the batch order
    x2, _ = get_batch(tokens, 4, 16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(x1, x2)
