import json

import torch

from conftest import tiny_model
from tiny_frontier.config import Config
from tiny_frontier.generate import generate
from tiny_frontier.run_folder import RunFolder


def test_generate_with_cache_matches_without():
    # Greedy decoding is deterministic, so the KV-cache path and the
    # "re-run everything" path must produce the exact same tokens.
    model = tiny_model()
    prompt = torch.randint(0, 64, (2, 5))
    with_cache = generate(model, prompt, 15, temperature=0, use_cache=True)
    without = generate(model, prompt, 15, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)
    assert with_cache.shape == (2, 20)


def test_run_folder_writes_everything(tmp_path):
    cfg = Config(name="unit")
    run = RunFolder(cfg, root=tmp_path)
    history = [
        {"step": 1, "train_loss": 5.0, "val_loss": 5.1, "lr": 1e-3, "tokens_seen": 8192, "wall_s": 1.0},
        {"step": 2, "train_loss": 4.0, "val_loss": 4.2, "lr": 1e-3, "tokens_seen": 16384, "wall_s": 2.0},
    ]
    for row in history:
        run.log(row)
    run.save_summary({"best_val_loss": 4.2})
    run.save_sample("Once upon a time")
    run.save_plot(history)
    run.save_model(tiny_model())

    for name in ["config.yaml", "git_commit.txt", "log.csv", "summary.json", "sample.txt", "loss.png", "model.pt"]:
        assert (run.path / name).exists(), name
    assert len((run.path / "log.csv").read_text().strip().splitlines()) == 3  # header + 2 rows
    assert json.loads((run.path / "summary.json").read_text())["best_val_loss"] == 4.2
