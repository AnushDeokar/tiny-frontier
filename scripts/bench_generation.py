"""Experiment 1: generation speed and real KV-cache memory.

    poetry run python scripts/bench_generation.py            # control + every finished exp1 run (seed 1337)
    poetry run python scripts/bench_generation.py --tokens 256 4096

For each trained model: generate N tokens greedily with the (rolling) KV cache,
time it, and read how many bytes the cache actually holds at the end.
Run it when nothing else is using the GPU, or the timings are meaningless.

Note: the models were trained on 256-token chunks, so text past ~256 tokens is
not meaningful. This measures speed and memory, not quality.
"""

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tiny_frontier.config import load_config  # noqa: E402
from tiny_frontier.generate import generate  # noqa: E402
from tiny_frontier.model.model import LlamaModel  # noqa: E402
from tiny_frontier.training.train import pick_device  # noqa: E402
from tiny_frontier.utils import kv_cache_bytes  # noqa: E402


def sync(device):
    # GPU work is queued asynchronously; wait for it before reading the clock.
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def default_runs():
    runs = sorted((ROOT / "runs").glob("control_seed1337_*/model.pt"))[-1:]
    runs += sorted((ROOT / "runs").glob("exp1_*_seed1337_*/model.pt"))
    return [p.parent for p in runs]


def variant_name(cfg):
    m = cfg.model
    return "GGGGGG (control)" if m.layer_pattern is None else f"{m.layer_pattern} W={m.window}"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--tokens", type=int, nargs="+", default=[256, 4096])
    args = p.parse_args()

    device = pick_device("auto")
    rows = []
    for run in args.runs or default_runs():
        cfg = load_config(run / "config.yaml")
        model = LlamaModel(**asdict(cfg.model))
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        model.to(device).eval()
        prompt = torch.tensor([[0]], device=device)  # a single start token

        generate(model, prompt, 16, temperature=0)  # warm-up: first GPU calls are slow
        for n in args.tokens:
            cache = model.new_cache(rolling=True)
            sync(device)
            t0 = time.perf_counter()
            generate(model, prompt, n, temperature=0, kv_cache=cache)  # no eos: always n tokens
            sync(device)
            seconds = time.perf_counter() - t0
            context = n + prompt.shape[1] - 1  # tokens the cache has absorbed
            rows.append({
                "variant": variant_name(cfg),
                "new_tokens": n,
                "tokens_per_s": round(n / seconds, 1),
                "kv_bytes_measured": cache.nbytes(),
                "kv_bytes_formula": kv_cache_bytes(cfg.model, context),
                "run_dir": run.relative_to(ROOT).as_posix() if run.is_absolute() else run.as_posix(),
            })
            print(f"{rows[-1]['variant']:18} {n:>5} tokens  {rows[-1]['tokens_per_s']:>8.1f} tok/s  "
                  f"KV {cache.nbytes() / 1024:>8.0f} KB")

    out = ROOT / "results" / "exp1_generation.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
