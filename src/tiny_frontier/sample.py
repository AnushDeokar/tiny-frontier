"""Generate text from a trained run.

    python -m tiny_frontier.sample --run runs/control_seed1337_20261008-213000 --prompt "The little dog"
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_frontier.config import load_config
from tiny_frontier.data.tokenizer import BOS_ID, EOS_TOKEN, load_tokenizer
from tiny_frontier.generate import generate
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.training.train import pick_device


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate text from a trained run folder.")
    p.add_argument("--run", required=True, help="run folder containing config.yaml and model.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("-n", "--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, help="fix for reproducible samples")
    args = p.parse_args(argv)

    run = Path(args.run)
    # The run folder's own config.yaml rebuilds exactly the architecture that was trained.
    cfg = load_config(run / "config.yaml")
    device = pick_device(cfg.train.device)

    model = LlamaModel(**asdict(cfg.model))
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(device).eval()

    tokenizer = load_tokenizer(cfg.train.tokenizer_path)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    ids = tokenizer.encode(args.prompt).ids
    if cfg.train.bos:
        ids = [BOS_ID] + ids  # match training: a BOS model always saw <s> first
    prompt = torch.tensor([ids], device=device)
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    for i in range(args.num_samples):
        out = generate(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=eos_id,
            generator=generator,
        )
        print(f"--- sample {i + 1} ---")
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
