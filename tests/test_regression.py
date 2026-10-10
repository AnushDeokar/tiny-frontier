"""Regression test: the code changes must not change the trained control.

Every run folder stores the weights (model.pt) and the val loss measured
when training finished (summary.json). Re-evaluating those weights with
today's code must reproduce that number. If a new feature accidentally
changed the default (control) code path, this number moves.

Skipped when no trained control run or val.bin is present (e.g. a fresh clone:
weights are not in git).
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from tiny_frontier.config import load_config
from tiny_frontier.data.loader import load_tokens
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.training.train import evaluate, pick_device

ROOT = Path(__file__).resolve().parents[1]
RUNS = sorted(p.parent for p in ROOT.glob("runs/control_seed*/model.pt"))
VAL = ROOT / "data" / "val.bin"

pytestmark = pytest.mark.skipif(
    not RUNS or not VAL.exists(), reason="needs a trained control run and data/val.bin"
)

# The stored number was measured on the training device (MPS) and rounded to
# 5 decimals. Re-measuring can differ in the last digits: float addition is
# not associative, so a different device or batch order of the sums changes
# the rounding. Any real code change moves the loss by far more than this.
TOLERANCE = 2e-4


def test_trained_control_still_gives_its_recorded_val_loss():
    run = RUNS[0]
    cfg = load_config(run / "config.yaml")
    cfg.train.val_path = str(VAL)
    recorded = json.loads((run / "summary.json").read_text())["final_val_loss"]

    model = LlamaModel(**asdict(cfg.model))
    # strict=True (the default): every saved tensor must have a matching place
    # in today's model and vice versa. A renamed or added parameter fails here.
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))

    device = pick_device("auto")
    model.to(device)
    val_loss = evaluate(model, load_tokens(cfg.train.val_path), cfg, device)

    assert abs(val_loss - recorded) < TOLERANCE, (
        f"{run.name}: recorded val loss {recorded:.5f}, today's code gives {val_loss:.5f}"
    )
