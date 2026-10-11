"""Every run leaves a folder with everything needed to understand it later.

    runs/{name}_seed{seed}_{timestamp}/
        config.yaml           exact settings used
        git_commit.txt        code version ("-dirty" = uncommitted changes)
        data_fingerprint.json hashes of train.bin / val.bin
        log.csv               one row per evaluation, written as training goes
        sample.txt            text generated from a fixed prompt
        loss_by_position.csv  mean val loss at each position of the context
        summary.json          headline numbers
        loss.png              train and val loss curves
        model.pt              final weights (gitignored; Experiment 2 re-uses them)
"""

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import yaml


def git_commit():
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_loss_by_position(run_dir, losses):
    with open(Path(run_dir) / "loss_by_position.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "loss"])
        for pos, loss in enumerate(losses.tolist()):
            writer.writerow([pos, round(loss, 5)])


class RunFolder:
    def __init__(self, cfg, root="runs"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / f"{cfg.name}_seed{cfg.train.seed}_{stamp}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.name = cfg.name

        (self.path / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        (self.path / "git_commit.txt").write_text(git_commit() + "\n")
        fingerprint = Path(cfg.train.train_path).parent / "fingerprint.json"
        if fingerprint.exists():
            shutil.copy(fingerprint, self.path / "data_fingerprint.json")

        self._fieldnames = None

    def log(self, row):
        """Append one row to log.csv immediately, so a crashed run still has its log."""
        first = self._fieldnames is None
        if first:
            self._fieldnames = list(row)
        with open(self.path / "log.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if first:
                writer.writeheader()
            writer.writerow(row)

    def save_summary(self, summary):
        with open(self.path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    def save_sample(self, text):
        (self.path / "sample.txt").write_text(text + "\n")

    def save_loss_by_position(self, losses):
        write_loss_by_position(self.path, losses)

    def save_model(self, model):
        torch.save(model.state_dict(), self.path / "model.pt")

    def save_plot(self, history):
        import matplotlib

        matplotlib.use("Agg")  # draw to a file, no window
        import matplotlib.pyplot as plt

        tokens_m = [h["tokens_seen"] / 1e6 for h in history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tokens_m, [h["train_loss"] for h in history], label="train", marker=".")
        ax.plot(tokens_m, [h["val_loss"] for h in history], label="val", marker=".")
        ax.set_xlabel("tokens seen (millions)")
        ax.set_ylabel("loss (nats/token)")
        ax.set_title(self.path.name)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path / "loss.png", dpi=120)
        plt.close(fig)
