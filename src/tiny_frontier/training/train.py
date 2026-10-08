"""Train one model from one config.

    python -m tiny_frontier.training.train --config configs/control.yaml --seed 1338
"""

import argparse
import random
import resource
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from tiny_frontier.config import load_config
from tiny_frontier.data.loader import get_batch, iter_eval_batches, load_tokens
from tiny_frontier.generate import generate
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.run_folder import RunFolder
from tiny_frontier.training.loss import lm_loss
from tiny_frontier.training.optim import build_optimizer, lr_at
from tiny_frontier.utils import kv_cache_bytes


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    # Seeds the init of the weights. (Batch order has its own generator below.)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def memory_mb(device):
    """Memory in use right now (we keep the max over the run as 'peak')."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    # CPU: peak resident memory of the process (bytes on macOS, KB on Linux).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def build_model(model_cfg):
    return LlamaModel(**asdict(model_cfg))


@torch.no_grad()
def evaluate(model, val_tokens, cfg, device):
    """Mean loss (nats/token) over the fixed validation slice."""
    model.eval()
    losses = []
    for x, y in iter_eval_batches(
        val_tokens, cfg.train.batch_size, cfg.model.max_seq_len, cfg.train.eval_batches
    ):
        x, y = x.to(device), y.to(device)
        losses.append(lm_loss(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def write_sample(model, cfg, device, max_new_tokens=200):
    from tiny_frontier.data.tokenizer import EOS_TOKEN, load_tokenizer

    tokenizer = load_tokenizer(cfg.train.tokenizer_path)
    prompt = torch.tensor([tokenizer.encode(cfg.train.sample_prompt).ids], device=device)
    out = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        eos_id=tokenizer.token_to_id(EOS_TOKEN),
        generator=torch.Generator().manual_seed(0),  # same sample every time for the same weights
    )
    return tokenizer.decode(out[0].tolist())


def train(cfg, run=None):
    t = cfg.train
    set_seed(t.seed)
    device = pick_device(t.device)

    train_tokens = load_tokens(t.train_path)
    val_tokens = load_tokens(t.val_path)

    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(model, t)
    data_gen = torch.Generator().manual_seed(t.seed)

    total_steps = cfg.total_steps
    print(f"run: {cfg.name} | seed {t.seed} | device {device}")
    if run is not None:
        print(f"run folder: {run.path}")
    print(
        f"params: {model.num_params():,} ({model.num_params(non_embedding=True):,} non-embedding)"
    )
    print(
        f"steps: {total_steps:,} x {cfg.tokens_per_step:,} tokens/step = "
        f"{total_steps * cfg.tokens_per_step:,} tokens"
    )

    history = []
    peak_mem = 0.0
    train_time = 0.0  # excludes evaluation, so tokens/sec measures training only
    t_start = time.perf_counter()
    t_last = t_start
    # Average train loss over every batch since the last eval: much less noisy
    # than the single last batch. Kept as a tensor so we don't force a GPU sync each step.
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    model.train()

    for step in range(total_steps):
        # The schedule is applied by hand each step; simpler to read than a torch LR scheduler.
        lr = lr_at(step, total_steps, t)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_tokens, t.batch_size, cfg.model.max_seq_len, generator=data_gen)
        x, y = x.to(device), y.to(device)

        loss = lm_loss(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clipping caps the total gradient norm at 1.0, so one unlucky batch
        # can't produce a giant update. The returned norm is the size *before*
        # clipping: a useful health signal (spikes = instability).
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()

        loss_sum += loss.detach()
        loss_count += 1

        is_last = step == total_steps - 1
        if step % t.log_every == 0 or is_last:
            # .item() forces the GPU to finish, so only call it when printing.
            print(
                f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"grad_norm {grad_norm.item():.2f}"
            )

        if (step + 1) % t.eval_every == 0 or is_last:
            now = time.perf_counter()
            train_time += now - t_last
            train_loss = (loss_sum / loss_count).item()
            loss_sum.zero_()
            loss_count = 0

            val_loss = evaluate(model, val_tokens, cfg, device)
            peak_mem = max(peak_mem, memory_mb(device))
            tokens_seen = (step + 1) * cfg.tokens_per_step
            row = {
                "step": step + 1,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "lr": lr,
                "tokens_seen": tokens_seen,
                "wall_s": round(time.perf_counter() - t_start, 1),
            }
            history.append(row)
            if run is not None:
                run.log(row)
            print(
                f"  eval @ step {step + 1}: train {train_loss:.4f} | val {val_loss:.4f} | "
                f"{tokens_seen / train_time:,.0f} train tokens/s"
            )
            t_last = time.perf_counter()

    tokens_seen = total_steps * cfg.tokens_per_step
    summary = {
        "name": cfg.name,
        "params": model.num_params(),
        "non_embedding_params": model.num_params(non_embedding=True),
        "tokens_seen": tokens_seen,
        "best_val_loss": min(h["val_loss"] for h in history),
        "final_val_loss": history[-1]["val_loss"],
        "wall_s": round(time.perf_counter() - t_start, 1),
        "tokens_per_s": round(tokens_seen / train_time),
        "peak_mem_mb": round(peak_mem, 1),
        # fp32 KV cache at the training context and at Experiment 2's stream length.
        "kv_bytes": {
            "256": kv_cache_bytes(cfg.model, 256),
            "4096": kv_cache_bytes(cfg.model, 4096),
        },
        "seed": t.seed,
        "device": device,
        "config": cfg.to_dict(),
    }

    print(
        f"done: best val {summary['best_val_loss']:.4f} | "
        f"{summary['wall_s'] / 60:.1f} min | {summary['tokens_per_s']:,} tokens/s"
    )

    if run is not None:
        sample = write_sample(model, cfg, device)
        print(f"sample:\n{sample}")
        run.save_sample(sample)
        run.save_summary(summary)
        run.save_plot(history)
        run.save_model(model)
        print(f"saved to {run.path}")

    return model, history, summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train one model from a YAML config.")
    p.add_argument("--config", required=True, help="path to a YAML config")
    # Command-line overrides, so a sweep or a seed repeat doesn't need a new YAML file.
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--total-tokens", type=int)
    p.add_argument("--name")
    p.add_argument("--no-save", action="store_true", help="don't create a run folder")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.total_tokens is not None:
        cfg.train.total_tokens = args.total_tokens
    if args.name is not None:
        cfg.name = args.name
    run = None if args.no_save else RunFolder(cfg)
    return train(cfg, run)


if __name__ == "__main__":
    main()
