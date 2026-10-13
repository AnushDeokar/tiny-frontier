"""Experiment 1 figures, built only from the run folders (no model, no GPU).

    poetry run python scripts/plot_exp1.py

Writes results/exp1_*.png and results/exp1_by_position.csv.
"""

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# (run name, label, pattern, window, colour). One colour per variant, used in every figure.
VARIANTS = [
    ("control", "GGGGGG (control)", None, None, "black"),
    ("exp1_LLLLLL_w128", "LLLLLL W=128", "LLLLLL", 128, "#9ecae1"),
    ("exp1_LLLLLL_w64", "LLLLLL W=64", "LLLLLL", 64, "#4292c6"),
    ("exp1_LLLLLL_w32", "LLLLLL W=32", "LLLLLL", 32, "#08519c"),
    ("exp1_LLLLLL_w16", "LLLLLL W=16", "LLLLLL", 16, "#08306b"),
    ("exp1_LLGLLG_w32", "LLGLLG W=32 (2 global)", "LLGLLG", 32, "#e6550d"),
    ("exp1_LLLLLG_w32", "LLLLLG W=32 (1 global)", "LLLLLG", 32, "#31a354"),
]
BUCKETS = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]


def runs(name):
    """Finished run folders of `name`, one per seed (latest wins)."""
    by_seed = {}
    for p in sorted(glob.glob(str(ROOT / f"runs/{name}_seed*_*/summary.json"))):
        by_seed[json.loads(Path(p).read_text())["seed"]] = Path(p).parent
    return list(by_seed.values())


def final_losses(name):
    return [json.loads((d / "summary.json").read_text())["final_val_loss"] for d in runs(name)]


def mean_curve(name, filename, key, x_key):
    curves = []
    for d in runs(name):
        with open(d / filename) as f:
            curves.append({int(r[x_key]): float(r[key]) for r in csv.DictReader(f)})
    xs = sorted(set.intersection(*[set(c) for c in curves]))
    return xs, [sum(c[x] for c in curves) / len(curves) for x in xs]


def kv_bytes(pattern, window, seq_len, n_layers=6, kv_heads=6, head_dim=32, bytes_per=4):
    layers = pattern or "G" * n_layers
    tokens = sum(min(seq_len, window) if c == "L" else seq_len for c in layers)
    return 2 * tokens * kv_heads * head_dim * bytes_per


def seed_label(n):
    return f"{n} seed" + ("s" if n > 1 else "")


def main():
    available = [v for v in VARIANTS if runs(v[0])]
    ctrl = final_losses("control")
    ctrl_mean = sum(ctrl) / len(ctrl)

    # 1. Val loss vs window size ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(min(ctrl), max(ctrl), color="black", alpha=0.08, label=f"control range ({seed_label(len(ctrl))})")
    ax.axhline(ctrl_mean, color="black", lw=1)
    local = [v for v in available if v[2] == "LLLLLL"]
    ws = [v[3] for v in local]
    means = [sum(final_losses(v[0])) / len(final_losses(v[0])) for v in local]
    ax.plot(ws, means, color="#08519c", marker="o", label="all local (LLLLLL)")
    for name, label, pattern, w, colour in available:
        losses = final_losses(name)
        if pattern is None:
            continue
        m = sum(losses) / len(losses)
        if len(losses) > 1:
            ax.vlines(w, min(losses), max(losses), color=colour, lw=3, alpha=0.6)
        if pattern != "LLLLLL":
            ax.plot([w], [m], marker="D", ms=8, color=colour, ls="none", label=f"{label}, {seed_label(len(losses))}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
    ax.set_xlabel("attention window W (tokens)")
    ax.set_ylabel("final val loss (nats/token)")
    ax.set_title("Experiment 1: val loss vs window (bars = seed range)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_loss_vs_window.png", dpi=150)
    plt.close(fig)

    # 2. Loss gap by position --------------------------------------------------
    _, c_curve = mean_curve("control", "loss_by_position.csv", "loss", "position")
    rows = []
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0, color="black", lw=1)
    centres = [((a + b) / 2) for a, b in BUCKETS]
    for name, label, pattern, w, colour in available[1:]:
        _, v = mean_curve(name, "loss_by_position.csv", "loss", "position")
        gaps = [sum(v[a:b + 1]) / (b - a + 1) - sum(c_curve[a:b + 1]) / (b - a + 1) for a, b in BUCKETS]
        ax.plot(centres, gaps, marker="o", color=colour, label=label)
        rows.append({"variant": label, **{f"pos_{a}_{b}": round(g, 4) for (a, b), g in zip(BUCKETS, gaps)}})
    ax.set_xscale("log", base=2)
    ax.set_xticks(centres, [f"{a}-{b}" for a, b in BUCKETS])
    ax.set_xlabel("tokens of context (position in the chunk)")
    ax.set_ylabel("val loss minus control  (below 0 = better)")
    ax.set_title("Where windows help and hurt")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_loss_by_position.png", dpi=150)
    plt.close(fig)
    with open(RESULTS / "exp1_by_position.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)

    # 3. Memory vs quality trade-off -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, label, pattern, w, colour in available:
        losses = final_losses(name)
        m = sum(losses) / len(losses)
        kb = kv_bytes(pattern, w, 4096) / 2**20
        ax.scatter([kb], [m], s=70, color=colour, zorder=3)
        # The two hybrids sit close together: put one label above, one below.
        offset = (8, -22) if name == "exp1_LLGLLG_w32" else (7, 4)
        ha = "left"
        ax.annotate(f"{label}\n({seed_label(len(losses))})", (kb, m), textcoords="offset points",
                    xytext=offset, fontsize=7.5, ha=ha)
    ax.set_xscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.015, hi)  # room for the label under the lowest point
    ax.set_xlabel("KV cache at 4,096 tokens (MB, fp32, log scale)")
    ax.set_ylabel("final val loss (nats/token)")
    ax.set_title("Memory vs quality: lower-left is better")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_memory_tradeoff.png", dpi=150)
    plt.close(fig)

    # 4. Gap vs control during training ----------------------------------------
    steps_c, c_val = mean_curve("control", "log.csv", "val_loss", "step")
    c_by_step = dict(zip(steps_c, c_val))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhline(0, color="black", lw=1)
    for name, label, pattern, w, colour in available[1:]:
        steps, v = mean_curve(name, "log.csv", "val_loss", "step")
        ax.plot([s * 8192 / 1e6 for s in steps], [x - c_by_step[s] for s, x in zip(steps, v)], color=colour, label=label)
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("val loss minus control  (below 0 = better)")
    ax.set_title("Gap to the control during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_gap_over_training.png", dpi=150)
    plt.close(fig)

    print("saved results/exp1_loss_vs_window.png, exp1_loss_by_position.png, "
          "exp1_memory_tradeoff.png, exp1_gap_over_training.png, exp1_by_position.csv")


if __name__ == "__main__":
    main()
