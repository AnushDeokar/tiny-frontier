"""Step 2: our implementation vs Hugging Face's on the real Gemma 3 270M weights.

    poetry run python scripts/gemma_parity.py

For each prompt: max |logit difference| over every position and all 262,144
vocabulary entries, how often both models agree on the top-1 next token, and
whether greedy generation produces the identical continuation.
Writes results/gemma_parity.csv.
"""

import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tiny_frontier.convert.gemma3 import load_gemma3  # noqa: E402
from tiny_frontier.generate import generate  # noqa: E402

CKPT = ROOT / "checkpoints" / "gemma-3-270m"
PROMPTS = [
    "The capital of France is",
    "Once upon a time, a little dragon",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The three primary colors are",
]
GEN_TOKENS = 30


@torch.no_grad()
def main():
    from transformers import AutoTokenizer, Gemma3ForCausalLM

    torch.manual_seed(0)
    t0 = time.perf_counter()
    ours, cfg = load_gemma3(CKPT)  # fp32 on CPU: the cleanest comparison
    hf = Gemma3ForCausalLM.from_pretrained(CKPT, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(CKPT)
    print(f"loaded both models in {time.perf_counter() - t0:.1f}s | params {ours.num_params():,}")

    # A prompt longer than the 512-token window: with the 5 short prompts the
    # local layers never cut anything off, so only this one tests the window.
    readme = (CKPT / "README.md").read_text()
    long_ids = tok(readme, return_tensors="pt").input_ids[:, :800]

    cases = [(p, tok(p, return_tensors="pt").input_ids) for p in PROMPTS]
    cases.append((f"<model card, first {long_ids.shape[1]} tokens>", long_ids))

    rows = []
    for name, ids in cases:
        a, b = ours(ids), hf(ids).logits
        diff = (a - b).abs().max().item()
        top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()

        greedy = "-"
        if ids.shape[1] < 100:  # generation only for the short prompts
            got = generate(ours, ids, max_new_tokens=GEN_TOKENS, temperature=0, eos_id=tok.eos_token_id)
            ref = hf.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False)
            # Hugging Face may also stop on extra end tokens from generation_config,
            # so compare the tokens both produced.
            n = min(got.shape[1], ref.shape[1])
            greedy = "identical" if torch.equal(got[:, :n], ref[:, :n]) else "DIFFERENT"
            text = tok.decode(got[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")
            print(f"\n{name!r} -> {text[:120]!r}")

        rows.append({
            "prompt": name,
            "tokens": ids.shape[1],
            "max_abs_logit_diff": f"{diff:.2e}",
            "top1_agreement": f"{top1:.4f}",
            "greedy_30_tokens": greedy,
        })

    print(f"\n{'prompt':42s} {'tokens':>6} {'max |diff|':>11} {'top-1 agree':>11} {'greedy':>10}")
    for r in rows:
        print(f"{r['prompt'][:42]:42s} {r['tokens']:>6} {r['max_abs_logit_diff']:>11} "
              f"{r['top1_agreement']:>11} {r['greedy_30_tokens']:>10}")

    out = ROOT / "results" / "gemma_parity.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
