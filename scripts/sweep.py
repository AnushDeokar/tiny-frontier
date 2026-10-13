"""Run a set of training runs one after another, then compare them.

LR sweep (1 seed, short runs, different learning rates):
    python scripts/sweep.py lr
    python scripts/sweep.py lr --lrs 1e-3 2e-3 3e-3 6e-3 --tokens 10000000

Seed repeats (same config, different seeds) once the LR is chosen:
    python scripts/sweep.py seeds
    python scripts/sweep.py seeds --seeds 1337 1338 1339

Experiment 1, local/global attention (control's config + one change each):
    python scripts/sweep.py exp1                                   # all 5 variants, seed 1337
    python scripts/sweep.py exp1 --seeds 1338 1339 --variants LLLLLL_w32

Runs that already finished (their summary.json exists) are skipped, so if
anything stops the sweep, just run the same command again.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
from tiny_frontier.config import load_config  # noqa: E402
from tiny_frontier.utils import kv_cache_bytes  # noqa: E402

# Experiment 1 variants: (layer pattern, attention window). The control
# (GGGGGG, all global) is not rerun: its 3 seeds from Step 1 are reused.
EXP1_VARIANTS = [
    ("LLLLLL", 16),
    ("LLLLLL", 32),
    ("LLLLLL", 64),
    ("LLLLLL", 128),
    ("LLGLLG", 32),
    # Added after the first results: Gemma's 5 local : 1 global ratio. Compared
    # with LLGLLG W=32 the only change is 1 global layer instead of 2.
    ("LLLLLG", 32),
]
# Noise floor from the control's 3 seeds was 0.0087 (max - min). A gap under
# NO_EFFECT is indistinguishable from seed noise; over CLEAR_EFFECT it can't be
# explained by noise; in between, more seeds are needed to decide.
NO_EFFECT, CLEAR_EFFECT = 0.01, 0.03


def lr_tag(lr):
    # 0.002 -> "2e-3", 0.0015 -> "1.5e-3": short, filename-safe and readable.
    mantissa, exponent = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def find_finished(name, seed):
    """The most recent finished run folder for (name, seed), or None.

    A folder without summary.json is a run that crashed or was stopped; it is ignored.
    """
    done = sorted(RUNS.glob(f"{name}_seed{seed}_*/summary.json"))
    return done[-1].parent if done else None


def run_job(job, config_path):
    cmd = [
        sys.executable, "-m", "tiny_frontier.training.train",
        "--config", str(config_path),
        "--name", job["name"],
        "--seed", str(job["seed"]),
        "--lr", str(job["lr"]),
        "--total-tokens", str(job["tokens"]),
        *job.get("extra", []),
    ]
    print(f"\n=== {job['name']} | seed {job['seed']} | lr {job['lr']} | {job['tokens']:,} tokens ===")
    # Each run is its own process: GPU memory is fully released between runs,
    # and one crashed run can't leave state behind for the next.
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_result(run_dir):
    summary = json.loads((run_dir / "summary.json").read_text())
    with open(run_dir / "log.csv") as f:
        log = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return summary, log


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path, series, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, log in series:
        ax.plot([r["tokens_seen"] / 1e6 for r in log], [r["val_loss"] for r in log], marker=".", label=label)
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("val loss (nats/token)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_lr_curve(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["lr"] for r in rows], [r["final_val_loss"] for r in rows], marker="o")
    ax.set_xscale("log")  # LRs are spread multiplicatively, so a log axis spaces them evenly
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final val loss (nats/token)")
    ax.set_title("LR sweep")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_table(rows, key_col):
    print(f"\n{key_col:>8} | {'best val':>8} | {'final val':>9} | {'tokens/s':>8} | {'time':>8}")
    print("-" * 54)
    for r in rows:
        print(
            f"{r[key_col]:>8} | {r['best_val_loss']:8.4f} | {r['final_val_loss']:9.4f} | "
            f"{r['tokens_per_s']:8,} | {r['wall_s'] / 60:6.1f} min"
        )


def finished_by_seed(name):
    """{seed: summary} for every finished run of `name`, the latest per seed."""
    out = {}
    for path in sorted(RUNS.glob(f"{name}_seed*_*/summary.json")):
        summary = json.loads(path.read_text())
        out[summary["seed"]] = summary  # sorted by timestamp, so the latest wins
    return out


def report_exp1(cfg):
    """Table of every Experiment 1 variant vs the control, from the run folders."""
    control = finished_by_seed(cfg.name)
    if not control:
        sys.exit(f"no finished '{cfg.name}' runs found to compare against")
    c_losses = [s["final_val_loss"] for s in control.values()]
    c_mean = sum(c_losses) / len(c_losses)

    rows = [{
        "variant": "GGGGGG (control)", "pattern": "GGGGGG", "window": "",
        "seeds": len(c_losses), "mean_val_loss": round(c_mean, 4),
        "min": min(c_losses), "max": max(c_losses), "delta_vs_control": 0.0,
        "verdict": "baseline",
        "kv_bytes_256": kv_cache_bytes(cfg.model, 256),
        "kv_bytes_4096": kv_cache_bytes(cfg.model, 4096),
    }]
    for pattern, window in EXP1_VARIANTS:
        done = finished_by_seed(f"exp1_{pattern}_w{window}")
        if not done:
            continue
        losses = [s["final_val_loss"] for s in done.values()]
        mean = sum(losses) / len(losses)
        delta = mean - c_mean
        if abs(delta) < NO_EFFECT:
            verdict = "no detectable effect"
        elif abs(delta) > CLEAR_EFFECT:
            verdict = "clear effect"
        else:
            verdict = "UNCLEAR: add seeds" if len(losses) < 3 else "small effect (3 seeds)"
        model_cfg = cfg.model.__class__(**{**cfg.model.__dict__, "layer_pattern": pattern, "window": window})
        rows.append({
            "variant": f"{pattern} W={window}", "pattern": pattern, "window": window,
            "seeds": len(losses), "mean_val_loss": round(mean, 4),
            "min": min(losses), "max": max(losses), "delta_vs_control": round(delta, 4),
            "verdict": verdict,
            "kv_bytes_256": kv_cache_bytes(model_cfg, 256),
            "kv_bytes_4096": kv_cache_bytes(model_cfg, 4096),
        })

    print(f"\n{'variant':18} {'seeds':>5} {'val loss':>9} {'vs control':>11} "
          f"{'KV@256':>9} {'KV@4096':>9}  verdict")
    print("-" * 86)
    for r in rows:
        print(f"{r['variant']:18} {r['seeds']:>5} {r['mean_val_loss']:9.4f} {r['delta_vs_control']:+11.4f} "
              f"{r['kv_bytes_256'] / 1024:7.0f}KB {r['kv_bytes_4096'] / 1024:7.0f}KB  {r['verdict']}")
    print(f"\nverdicts: |gap| < {NO_EFFECT} = noise level, > {CLEAR_EFFECT} = clear; in between, add seeds.")
    write_csv(RESULTS / "exp1.csv", rows)
    print("saved results/exp1.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["lr", "seeds", "exp1"])
    p.add_argument("--config", default="configs/control.yaml")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 2e-3, 3e-3, 6e-3])
    p.add_argument("--seeds", type=int, nargs="+",
                   help="default: 1337 1338 1339 for 'seeds', 1337 for 'exp1'")
    p.add_argument("--dry-run", action="store_true", help="show the plan (and current results) without training")
    p.add_argument("--variants", nargs="+",
                   help="exp1 only: subset of variants, e.g. LLLLLL_w32 LLGLLG_w32")
    p.add_argument("--seed", type=int, default=1337, help="seed for the LR sweep")
    p.add_argument("--lr", type=float, help="LR for seed runs (default: the config's)")
    p.add_argument("--tokens", type=int, help="token budget (default: 10M for lr, config's for seeds)")
    args = p.parse_args()

    config_path = ROOT / args.config
    cfg = load_config(config_path)

    if args.mode == "lr":
        tokens = args.tokens or 10_000_000
        jobs = [
            {"name": f"sweep_lr{lr_tag(lr)}", "seed": args.seed, "lr": lr, "tokens": tokens, "key": lr_tag(lr)}
            for lr in sorted(args.lrs)
        ]
    elif args.mode == "seeds":
        lr = args.lr or cfg.train.lr
        tokens = args.tokens or cfg.train.total_tokens
        jobs = [
            {"name": cfg.name, "seed": s, "lr": lr, "tokens": tokens, "key": str(s)}
            for s in (args.seeds or [1337, 1338, 1339])
        ]
    else:  # exp1
        lr = args.lr or cfg.train.lr
        tokens = args.tokens or cfg.train.total_tokens
        variants = [(p_, w) for p_, w in EXP1_VARIANTS
                    if not args.variants or f"{p_}_w{w}" in args.variants]
        if not variants:
            sys.exit(f"no variant matches {args.variants}; choose from "
                     f"{[f'{p_}_w{w}' for p_, w in EXP1_VARIANTS]}")
        jobs = [
            {
                "name": f"exp1_{p_}_w{w}", "seed": s, "lr": lr, "tokens": tokens,
                "key": f"{p_}_w{w}",
                "extra": ["--pattern", p_, "--window", str(w)],
            }
            for p_, w in variants
            for s in (args.seeds or [1337])
        ]

    # Plan: which runs are already done, which still need to run.
    todo = [j for j in jobs if find_finished(j["name"], j["seed"]) is None]
    print(f"{len(jobs)} runs in this sweep, {len(jobs) - len(todo)} already finished, {len(todo)} to run.")
    for j in jobs:
        print(f"  {'done ' if j not in todo else 'to do'}  {j['name']} seed {j['seed']} lr {j['lr']}")

    if args.dry_run:
        print(f"\ndry run: nothing trained. {len(todo)} runs x ~27 min = ~{len(todo) * 27 / 60:.1f} h")
        if args.mode == "exp1":
            report_exp1(cfg)
        return

    t0 = time.perf_counter()
    for job in todo:
        run_job(job, config_path)
    if todo:
        print(f"\nsweep finished in {(time.perf_counter() - t0) / 60:.1f} min")

    if args.mode == "exp1":
        report_exp1(cfg)
        return

    # Collect results from the run folders (not from memory), so this also
    # works when every run was done in an earlier session.
    rows, series = [], []
    for j in jobs:
        run_dir = find_finished(j["name"], j["seed"])
        summary, log = load_result(run_dir)
        rows.append(
            {
                ("lr" if args.mode == "lr" else "seed"): j["lr"] if args.mode == "lr" else j["seed"],
                "key": j["key"],
                "best_val_loss": summary["best_val_loss"],
                "final_val_loss": summary["final_val_loss"],
                "tokens_seen": summary["tokens_seen"],
                "tokens_per_s": summary["tokens_per_s"],
                "wall_s": summary["wall_s"],
                "run_dir": run_dir.relative_to(ROOT).as_posix(),
            }
        )
        series.append((j["key"], log))

    RESULTS.mkdir(exist_ok=True)

    if args.mode == "lr":
        print_table(rows, "key")
        write_csv(RESULTS / "lr_sweep.csv", rows)
        plot_lr_curve(RESULTS / "lr_sweep.png", rows)
        plot_curves(RESULTS / "lr_sweep_curves.png", [(f"lr {k}", log) for k, log in series], "LR sweep: val loss")

        best = min(rows, key=lambda r: r["final_val_loss"])
        print(f"\nbest LR: {best['key']} (final val {best['final_val_loss']:.4f})")
        lrs = [r["lr"] for r in rows]
        if best["lr"] in (min(lrs), max(lrs)):
            side = "lower" if best["lr"] == min(lrs) else "higher"
            print(f"WARNING: best LR is at the edge of the range. Add a {side} LR and re-run (finished runs are skipped).")
        close = [r for r in rows if r is not best and r["final_val_loss"] - best["final_val_loss"] < 0.01]
        if close:
            names = ", ".join(r["key"] for r in close)
            print(
                f"NOTE: {names} within 0.01 of the best: about noise level. "
                "Prefer the lower LR for the longer 50M-token runs."
            )
        print(f"\nsaved results/lr_sweep.csv, results/lr_sweep.png, results/lr_sweep_curves.png")
    else:
        print_table(rows, "key")
        finals = [r["final_val_loss"] for r in rows]
        mean = sum(finals) / len(finals)
        print(
            f"\nfinal val loss over {len(finals)} seeds: mean {mean:.4f} | "
            f"min {min(finals):.4f} | max {max(finals):.4f} | spread (max-min) {max(finals) - min(finals):.4f}"
        )
        print("Any effect you claim later should be clearly bigger than this spread.")
        write_csv(RESULTS / f"{cfg.name}_seeds.csv", rows)
        plot_curves(RESULTS / f"{cfg.name}_seeds.png", [(f"seed {k}", log) for k, log in series], f"{cfg.name}: seeds")
        print(f"saved results/{cfg.name}_seeds.csv, results/{cfg.name}_seeds.png")


if __name__ == "__main__":
    main()
