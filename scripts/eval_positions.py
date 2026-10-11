"""Compute loss_by_position.csv for finished runs that don't have it yet.

    poetry run python scripts/eval_positions.py                # every run missing it
    poetry run python scripts/eval_positions.py runs/control_* # specific runs

Uses each run's saved model.pt and config.yaml, so nothing is retrained.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tiny_frontier.config import load_config  # noqa: E402
from tiny_frontier.data.loader import load_tokens  # noqa: E402
from tiny_frontier.evaluation import loss_by_position, max_eval_batches  # noqa: E402
from tiny_frontier.model.model import LlamaModel  # noqa: E402
from tiny_frontier.run_folder import write_loss_by_position  # noqa: E402
from tiny_frontier.training.train import pick_device  # noqa: E402


def main():
    if len(sys.argv) > 1:
        runs = [Path(a) for a in sys.argv[1:]]
    else:
        runs = sorted(p.parent for p in (ROOT / "runs").glob("*/model.pt")
                      if not (p.parent / "loss_by_position.csv").exists())
    if not runs:
        print("nothing to do: every run with weights already has loss_by_position.csv")
        return

    device = pick_device("auto")
    val = load_tokens(ROOT / "data" / "val.bin")
    for run in runs:
        cfg = load_config(run / "config.yaml")
        model = LlamaModel(**asdict(cfg.model))
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        model.to(device)
        T, B = cfg.model.max_seq_len, cfg.train.batch_size
        losses = loss_by_position(model, val, B, T, max_eval_batches(val, B, T), device)
        write_loss_by_position(run, losses)
        print(f"{run.name}: position 0 {losses[0]:.3f} | position {T - 1} {losses[-1]:.3f} | mean {losses.mean():.4f}")


if __name__ == "__main__":
    main()
