"""Experiment 2: streaming past the training length with different KV caches.

    poetry run python scripts/exp2_stream.py              # evaluate everything + plots (~10 min)
    poetry run python scripts/exp2_stream.py --plot-only  # redraw from the saved CSVs

Cases (all on the same 32 streams of 4,096 validation tokens, fed one token
at a time):
  a  control (dense)            full cache       grows forever; positions > 256 never seen in training
  b  control (dense)            rolling(64)      first tokens evicted: the StreamingLLM failure
  c  control (dense)            sink(4, 64)      StreamingLLM fix: keep the first 4 tokens
  d  control + learned sink     rolling(64)      gpt-oss fix: trained-in sink
  e  LLLLLL W=64 (Exp 1)        rolling(64)      trained with the window: nothing to lose
  f  control + BOS              rolling(64)      <s> at position 0 in training: does evicting it now hurt?
  g  control + BOS              sink(4, 64)      ...and does keeping it fix that?

BOS models get streams that start with <s>, exactly like their training chunks.

    poetry run python scripts/exp2_stream.py --cases f,g  # only these; merges into the saved CSVs
"""

import argparse
import csv
import glob
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tiny_frontier.config import load_config  # noqa: E402
from tiny_frontier.data.loader import load_tokens  # noqa: E402
from tiny_frontier.data.tokenizer import BOS_ID  # noqa: E402
from tiny_frontier.evaluation import first_token_attention, make_streams, stream_losses  # noqa: E402
from tiny_frontier.model.attention import KVCache, StreamCache  # noqa: E402
from tiny_frontier.model.model import LlamaModel  # noqa: E402
from tiny_frontier.training.train import pick_device  # noqa: E402

RESULTS = ROOT / "results"
N_STREAMS, STREAM_LEN, BUCKET, W, N_SINK = 32, 4096, 128, 64, 4

# (id, label, run name, cache policy, colour)
CASES = [
    ("a", "a  dense, full cache", "control", "full", "black"),
    ("b", "b  dense, rolling(64)", "control", "rolling", "#d62728"),
    ("c", "c  dense, sink(4, 64)", "control", "sink", "#1f77b4"),
    ("d", "d  learned sink, rolling(64)", "control_sink", "rolling", "#2ca02c"),
    ("e", "e  trained with W=64, rolling(64)", "exp1_LLLLLL_w64", "rolling", "#9467bd"),
    ("f", "f  BOS-trained, rolling(64)", "control_bos", "rolling", "#ff7f0e"),
    ("g", "g  BOS-trained, sink(4, 64)", "control_bos", "sink", "#8c564b"),
]
ATTENTION_MODELS = ["control", "control_sink", "exp1_LLLLLL_w64", "control_bos"]


def run_dirs(name):
    """Finished run folders for `name` (one per seed), weights included."""
    return sorted(Path(p).parent for p in glob.glob(str(ROOT / f"runs/{name}_seed*_*/model.pt")))


def load_model(run, device):
    cfg = load_config(run / "config.yaml")
    model = LlamaModel(**asdict(cfg.model))
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.to(device).eval(), cfg


def make_cache(model, policy):
    n = len(model.blocks)
    if policy == "full":
        # Nothing is ever evicted, so text positions = slot positions; the
        # plain cache is identical (tested) and much faster here.
        return KVCache(n)
    if policy == "rolling":
        return StreamCache(n, window=W)
    if policy == "sink":
        return StreamCache(n, n_sink=N_SINK, window=W)
    raise ValueError(policy)


def evaluate(device, attention_only=False, only=None):
    val = load_tokens(ROOT / "data" / "val.bin")
    if not attention_only:
        evaluate_streams(device, val, only)
    evaluate_attention(device, val)


def bos_stream_losses(model, streams, cache, device):
    """Feed [<s>, t0, t1, ...] (as in every BOS training chunk), then drop the
    loss for predicting t0 from <s> alone. What's left lines up exactly with
    the other cases: entry t = loss for predicting t_{t+1} after t_0..t_t."""
    bos = torch.full((streams.shape[0], 1), BOS_ID, dtype=streams.dtype)
    return stream_losses(model, torch.cat([bos, streams], dim=1), cache, device)[:, 1:]


def evaluate_streams(device, val, only=None):
    streams = make_streams(val, N_STREAMS, STREAM_LEN)
    per_position, buckets = [], []
    if only:
        # Keep the saved rows of every other case; recompute just `only`.
        per_position = [r for r in read_csv(RESULTS / "exp2_stream_positions.csv") if r["case"] not in only]
        buckets = [r for r in read_csv(RESULTS / "exp2_stream.csv") if r["case"] not in only]
    for case_id, label, name, policy, _ in CASES:
        if only and case_id not in only:
            continue
        runs = run_dirs(name)
        if not runs:
            print(f"skip {case_id}: no finished '{name}' runs")
            continue
        losses = []  # [streams from all seeds, positions]
        for run in runs:
            model, cfg = load_model(run, device)
            score = bos_stream_losses if cfg.train.bos else stream_losses
            losses.append(score(model, streams, make_cache(model, policy), device))
            print(f"{case_id} {run.name}: mean {losses[-1].mean():.3f}")
        losses = torch.cat(losses)  # [n_seeds * N_STREAMS, STREAM_LEN - 1]
        mean = losses.mean(0)
        for t, m in enumerate(mean.tolist()):
            per_position.append({"case": case_id, "position": t + 1, "loss": round(m, 4)})
        for start in range(0, STREAM_LEN, BUCKET):
            # position p = predicting token p; bucket [start, start + BUCKET)
            cols = losses[:, max(start - 1, 0) : start + BUCKET - 1]
            per_stream = cols.mean(1)  # one value per stream: error bar across streams
            buckets.append({
                "case": case_id, "label": label, "bucket_start": start, "bucket_end": start + BUCKET - 1,
                "mean_loss": round(per_stream.mean().item(), 4),
                "sem": round((per_stream.std() / per_stream.numel() ** 0.5).item(), 4),
                "n_runs": len(runs),
            })
    write_csv(RESULTS / "exp2_stream_positions.csv", per_position)
    write_csv(RESULTS / "exp2_stream.csv", buckets)


def evaluate_attention(device, val):
    # Attention on token 0, per layer and head, on normal 256-token chunks.
    rows = []
    for name in ATTENTION_MODELS:
        for run in run_dirs(name):
            model, cfg = load_model(run, device)
            bos = BOS_ID if cfg.train.bos else None  # for a BOS model, "token 0" IS <s>
            mass, sink = first_token_attention(model, val, cfg.train.batch_size, cfg.model.max_seq_len, 8, device,
                                               bos_id=bos)
            for layer in range(mass.shape[0]):
                for head in range(mass.shape[1]):
                    rows.append({
                        "run": run.name, "model": name, "layer": layer, "head": head,
                        "attn_on_token0": round(mass[layer, head].item(), 4),
                        "learned_sink_share": "" if sink is None else round(sink[layer, head].item(), 4),
                    })
    write_csv(RESULTS / "exp2_first_token_attention.csv", rows)


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


MIN_QUERY, BLOCK = 16, 256  # must match first_token_attention's defaults


def even_share(window):
    """Weight token 0 would get if every head spread attention evenly.

    Query q sees min(q+1, window) keys; token 0 is one of them only while
    q < window. Averaged over the queries we measure (16..255).
    """
    window = window or BLOCK
    qs = range(MIN_QUERY, BLOCK)
    return sum((1 / min(q + 1, window)) if q < window else 0.0 for q in qs) / len(qs)


def window_of(model_name):
    m = re.search(r"_w(\d+)", model_name)
    return int(m.group(1)) if m else None


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # ---- Hero chart: left = whole stream (log x), right = zoom past 256 ----
    pos = read_csv(RESULTS / "exp2_stream_positions.csv")
    buckets = read_csv(RESULTS / "exp2_stream.csv")
    fig, (ax, zoom, cost) = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for case_id, label, _, _, colour in CASES:
        ys = [float(r["loss"]) for r in pos if r["case"] == case_id]
        if not ys:
            continue
        k = 32
        smooth = [sum(ys[max(0, i - k + 1) : i + 1]) / len(ys[max(0, i - k + 1) : i + 1]) for i in range(len(ys))]
        ax.plot(range(1, len(ys) + 1), smooth, color=colour, label=label, lw=1.6)
    ax.axvline(W, color="grey", ls=":", lw=1)
    ax.axvline(256, color="grey", ls="--", lw=1)
    top = ax.get_ylim()[1]
    ax.text(W * 0.95, top * 0.98, "cache\nwindow (64)", fontsize=8, color="grey", va="top", ha="right")
    ax.text(256 * 0.95, top * 0.98, "training\nlength (256)", fontsize=8, color="grey", va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("position in the stream (tokens, log scale)")
    ax.set_ylabel("loss (nats/token, 32-token moving avg)")
    ax.set_title("Whole stream: only the full cache breaks", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    # Zoom: 128-token bucket means with error bars (SEM across streams), a left out.
    offsets = {"b": -24, "c": -14, "d": -4, "e": 6, "f": 16, "g": 26}  # nudge sideways so bars don't hide each other
    for case_id, label, _, _, colour in CASES:
        if case_id == "a":
            continue
        rows = [r for r in buckets if r["case"] == case_id and int(r["bucket_start"]) >= 256]
        if not rows:
            continue
        xs = [int(r["bucket_start"]) + BUCKET / 2 + offsets[case_id] for r in rows]
        zoom.errorbar(xs, [float(r["mean_loss"]) for r in rows], yerr=[float(r["sem"]) for r in rows],
                      color=colour, marker="o", ms=3, lw=1.2, capsize=2, label=label)
    zoom.set_xlabel("position in the stream (tokens; 128-token buckets)")
    zoom.set_ylabel("mean loss ± SEM")
    zoom.set_title("Past the training length (case a off the scale, ~4.1–4.8)", fontsize=10)
    zoom.grid(alpha=0.3)

    # Paired cost of evicting token 0: same model, same streams, only the cache
    # differs (rolling minus sink). Seed and text noise cancel out.
    by = {(r["case"], int(r["bucket_start"])): float(r["mean_loss"]) for r in buckets}
    for rolling, sink, colour, name in [("b", "c", "#d62728", "dense control (no BOS)"),
                                        ("f", "g", "#ff7f0e", "BOS-trained control")]:
        starts = [s for s in range(0, STREAM_LEN, BUCKET) if (rolling, s) in by and (sink, s) in by]
        if not starts:
            continue
        gaps = [by[(rolling, s)] - by[(sink, s)] for s in starts]
        mean_gap = sum(gaps) / len(gaps)
        cost.plot([s + BUCKET / 2 for s in starts], gaps, color=colour, marker="o", ms=3, lw=1.4,
                  label=f"{name}: mean {mean_gap:+.3f}, worse in {sum(g > 0 for g in gaps)}/{len(gaps)} buckets")
    cost.axhline(0, color="grey", lw=1)
    cost.set_xlabel("position in the stream (tokens; 128-token buckets)")
    cost.set_ylabel("loss(rolling 64) − loss(sink 4 + 64)")
    cost.set_title("Cost of evicting the first tokens (paired)", fontsize=10)
    cost.legend(fontsize=7.5, loc="upper left")
    cost.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Streaming 4,096 tokens with a 64-token cache: evicting token 0 only hurts when "
                 "training put a fixed token there", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(RESULTS / "exp2_stream.png", dpi=150)
    plt.close(fig)

    # ---- Heatmaps: token-0 attention relative to an even spread, + learned-sink share ----
    att = read_csv(RESULTS / "exp2_first_token_attention.csv")

    def mean_grid(model_name, column):
        rows = [r for r in att if r["model"] == model_name and r[column] != ""]
        if not rows:
            return None, 0
        L = max(int(r["layer"]) for r in rows) + 1
        H = max(int(r["head"]) for r in rows) + 1
        n_seeds = len({r["run"] for r in rows})
        grid = [[0.0] * H for _ in range(L)]
        for r in rows:  # average over seeds
            grid[int(r["layer"])][int(r["head"])] += float(r[column]) / n_seeds
        return grid, n_seeds

    panels = []
    for model_name in ATTENTION_MODELS:
        grid, n = mean_grid(model_name, "attn_on_token0")
        if grid:
            base = even_share(window_of(model_name))
            ratio = [[v / base for v in row] for row in grid]
            panels.append(("ratio", f"{model_name}: token 0 vs even spread\n({n} seed{'s' * (n > 1)}; 1.0 = even = {base:.4f})", ratio))
    grid, n = mean_grid("control_sink", "learned_sink_share")
    if grid:
        panels.append(("share", f"control_sink: weight on the learned sink\n({n} seed)", grid))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.0), squeeze=False)
    ratio_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=4)  # blue = ignored, white = even, red = sink
    for ax, (kind, title, grid) in zip(axes[0], panels):
        L, H = len(grid), len(grid[0])
        if kind == "ratio":
            im_ratio = ax.imshow(grid, norm=ratio_norm, cmap="RdBu_r", aspect="auto")
            fmt, dark = "{:.2f}", lambda v: v > 3 or v < 0.25
        else:
            im_share = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            fmt, dark = "{:.2f}", lambda v: v < 0.6
        for i in range(L):
            for j in range(H):
                ax.text(j, i, fmt.format(grid[i][j]), ha="center", va="center", fontsize=7,
                        color="white" if dark(grid[i][j]) else "black")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(H))
        ax.set_yticks(range(L))
    ratio_axes = [a for a, p in zip(axes[0], panels) if p[0] == "ratio"]
    fig.colorbar(im_ratio, ax=ratio_axes, shrink=0.8, location="bottom", pad=0.18,
                 label="attention on token 0 ÷ even-spread share  (>1 = sink, <1 = ignored)")
    if any(p[0] == "share" for p in panels):
        fig.colorbar(im_share, ax=axes[0][-1], shrink=0.8, location="bottom", pad=0.18,
                     label="share of attention to 'nothing'")
    fig.suptitle(f"Where spare attention goes (queries at positions {MIN_QUERY}-{BLOCK - 1}, 256-token chunks)",
                 fontsize=10, y=1.02)
    fig.savefig(RESULTS / "exp2_first_token_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_post()


def plot_post():
    """Single-message figures for the write-up (narrower than the full panels above)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    colours = {c[0]: c[4] for c in CASES}
    labels = {"a": "full cache (keeps every token)", "b": "rolling cache, last 64 tokens",
              "e": "trained with a 64-token window, rolling cache"}

    # 1. Only the full cache breaks.
    pos = read_csv(RESULTS / "exp2_stream_positions.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for case_id in ["a", "b", "e"]:
        ys = [float(r["loss"]) for r in pos if r["case"] == case_id]
        if not ys:
            continue
        k = 32
        smooth = [sum(ys[max(0, i - k + 1) : i + 1]) / len(ys[max(0, i - k + 1) : i + 1]) for i in range(len(ys))]
        ax.plot(range(1, len(ys) + 1), smooth, color=colours[case_id], label=labels[case_id], lw=1.8)
    ax.axvline(256, color="grey", ls="--", lw=1)
    ax.text(256 * 1.06, 4.9, "training length (256)", fontsize=8.5, color="grey", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("position in the stream (tokens, log scale)")
    ax.set_ylabel("loss (nats/token, 32-token moving avg)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="center left")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_post_stream.png", dpi=150)
    plt.close(fig)

    # 2. Paired cost of evicting the first tokens.
    buckets = read_csv(RESULTS / "exp2_stream.csv")
    by = {(r["case"], int(r["bucket_start"])): float(r["mean_loss"]) for r in buckets}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for rolling, sink, colour, name in [("b", "c", "#d62728", "control (chunks start mid-text)"),
                                        ("f", "g", "#ff7f0e", "control + <s> at the start of every chunk")]:
        starts = [s for s in range(0, STREAM_LEN, BUCKET) if (rolling, s) in by and (sink, s) in by]
        if not starts:
            continue
        gaps = [by[(rolling, s)] - by[(sink, s)] for s in starts]
        ax.plot([s + BUCKET / 2 for s in starts], gaps, color=colour, marker="o", ms=3.5, lw=1.6,
                label=f"{name}: mean {sum(gaps) / len(gaps):+.3f}")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("position in the stream (tokens; 128-token buckets)")
    ax.set_ylabel("extra loss from evicting the first tokens")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_post_eviction_cost.png", dpi=150)
    plt.close(fig)

    # 3. Where attention goes: control vs BOS (vs even spread), and the learned sink.
    att = read_csv(RESULTS / "exp2_first_token_attention.csv")

    def mean_grid(model_name, column):
        rows = [r for r in att if r["model"] == model_name and r[column] != ""]
        if not rows:
            return None
        n_seeds = len({r["run"] for r in rows})
        grid = [[0.0] * 6 for _ in range(6)]
        for r in rows:
            grid[int(r["layer"])][int(r["head"])] += float(r[column]) / n_seeds
        return grid

    base = even_share(None)
    panels = []
    for name, title in [("control", "control: chunks start mid-text"),
                        ("control_bos", "control + <s> at position 0")]:
        g = mean_grid(name, "attn_on_token0")
        if g:
            panels.append(("ratio", title, [[v / base for v in row] for row in g]))
    g = mean_grid("control_sink", "learned_sink_share")
    if g:
        panels.append(("share", "learned sink: share sent to 'nothing'", g))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.3), squeeze=False)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=4)
    for ax, (kind, title, grid) in zip(axes[0], panels):
        if kind == "ratio":
            im_r = ax.imshow(grid, norm=norm, cmap="RdBu_r", aspect="auto")
            fmt = lambda v: f"{v:.0f}×" if v >= 9.5 else f"{v:.1f}×"
            dark = lambda v: v > 3 or v < 0.25
        else:
            im_s = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            fmt = lambda v: f"{v * 100:.0f}%"
            dark = lambda v: v < 0.6
        for i in range(6):
            for j in range(6):
                ax.text(j, i, fmt(grid[i][j]), ha="center", va="center", fontsize=7.5,
                        color="white" if dark(grid[i][j]) else "black")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
    ratio_axes = [a for a, p in zip(axes[0], panels) if p[0] == "ratio"]
    fig.colorbar(im_r, ax=ratio_axes, shrink=0.8, location="bottom", pad=0.2,
                 label="attention on token 0 ÷ an even share  (1× = even, >1× = sink)")
    if panels and panels[-1][0] == "share":
        fig.colorbar(im_s, ax=axes[0][-1], shrink=0.8, location="bottom", pad=0.2, label="share of attention")
    fig.savefig(RESULTS / "exp2_post_attention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary():
    rows = read_csv(RESULTS / "exp2_stream.csv")
    show = [0, 128, 256, 512, 1024, 2048, 3968]
    print(f"\n{'case':38}" + "".join(f"{s:>8}" for s in show))
    for case_id, label, _, _, _ in CASES:
        r = {int(x["bucket_start"]): x for x in rows if x["case"] == case_id}
        if r:
            print(f"{label:38}" + "".join(f"{float(r[s]['mean_loss']):8.3f}" for s in show))
    print("(columns: start of each 128-token bucket; values: mean loss)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--attention-only", action="store_true", help="redo only the token-0 attention part")
    p.add_argument("--cases", help="comma-separated case ids to (re)compute, e.g. f,g; others are kept from the CSVs")
    args = p.parse_args()
    only = set(args.cases.split(",")) if args.cases else None
    if not args.plot_only:
        evaluate(pick_device("auto"), attention_only=args.attention_only, only=only)
    plot()
    summary()
    print("saved results/exp2_stream.csv, exp2_stream_positions.csv, exp2_first_token_attention.csv, "
          "exp2_stream.png, exp2_first_token_heatmap.png")


if __name__ == "__main__":
    main()
