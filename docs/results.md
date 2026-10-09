# Results log

Running notes, written as each step finishes (ground rule 8). The blog post is
written from this file. All numbers come from `runs/*/summary.json` and
`results/*.csv`; every run folder records its config, git commit and data hashes.

---

## Setup (shared by every experiment)

**Data.** TinyStories (`roneneldan/TinyStories`). Byte-level BPE tokenizer,
vocab 4,096, trained on the train split, with `</s>` appended after every story.
`train.bin` = first 100M tokens of the train split; `val.bin` = first 1M tokens
of the official validation split. SHA-256 hashes in `data/fingerprint.json`.

**Control model.** Llama-style decoder, written from scratch in PyTorch.

| Item | Value |
| --- | --- |
| Layers / width / heads | 6 / 192 / 6 (head dim 32) |
| Block | pre-RMSNorm → causal attention with RoPE → residual; pre-RMSNorm → SwiGLU (hidden 512) → residual |
| Embeddings | tied input/output, no biases |
| Init | N(0, 0.02); residual output projections scaled by 1/√(2·n_layers) |
| Params | 3,443,136 total, 2,656,704 non-embedding |
| Context | 256 tokens |

**Training.** AdamW (β = 0.9, 0.95; weight decay 0.1 on matrices only),
200-step linear warmup then cosine decay to 10% of peak, grad clip 1.0,
batch 32 × 256 = 8,192 tokens/step, 50M tokens (6,103 steps) per headline run.
Random training windows; batch order depends only on the seed.

**Evaluation.** Mean cross-entropy (nats/token) on a fixed slice of
491,520 validation tokens (60 batches of 32 × 256 non-overlapping windows),
identical for every run.

**Hardware.** MacBook Pro M4, PyTorch MPS backend, fp32.

**Tests.** 29 unit tests: RoPE (norm preservation, relative-position property),
RMSNorm vs reference, causality, KV-cached decoding = full forward
(MHA, GQA, MQA, custom head_dim), GQA = MHA with shared K/V weights,
overfitting one batch, exact parameter count, initial loss ≈ ln(4096).

---

## Step 1: control model

### LR sweep

Peak LR swept at **10M tokens, seed 1337** (each run is a complete short
schedule: warmup + cosine over 1,220 steps).

| Peak LR | Final val loss |
| --- | --- |
| 1e-3 | 2.4282 |
| 2e-3 | 2.3034 |
| 3e-3 | 2.2669 |
| **6e-3** | **2.2456** (best measured) |
| 1.2e-2 | 2.3832 |

![LR sweep](../results/lr_sweep.png)

- The curve is U-shaped on a log-LR axis, and **asymmetric**: halving the LR
  from 6e-3 costs +0.021, doubling it costs +0.138. Too high hurts far more
  than too low.
- A parabola through 3e-3 / 6e-3 / 1.2e-2 (log axis) puts the minimum near
  **4.7e-3**, about 0.01 below 6e-3 (noise level).
- **Chosen: 5e-3**, rounding toward the fitted minimum and toward the lower
  side, because (a) the U is asymmetric and (b) the headline runs are 5×
  longer, and longer runs tend to prefer a slightly lower LR.
- The LR choice moved the loss by up to 0.18 at 10M tokens, ~20× the seed
  noise below. An untuned baseline (the original 3e-4) would have been well
  below the swept range.
- Experiments 1 and 2 reuse this LR: their changes (attention windows, sinks)
  don't change the weight shapes. Experiment 3's S and L sizes get their own
  short sweeps.

### Control: 3 seeds at 50M tokens, LR 5e-3

| Seed | Final val loss | Train tokens/s | Wall-clock |
| --- | --- | --- | --- |
| 1337 | 1.8830 | 33,140 | 27.4 min |
| 1338 | 1.8743 | 33,677 | 26.9 min |
| 1339 | 1.8821 | 33,508 | 27.1 min |
| **Mean** | **1.8798** (min 1.8743, max 1.8830) | | |

![Control seeds](../results/control_seeds.png)

**Seed spread (max − min) = 0.0087.** This is the noise floor for every later
comparison: a difference under ~0.01 is not distinguishable from seed noise;
0.02 or more is a real effect (when it holds across seeds).

Perplexity e^1.88 ≈ 6.6: on average the model is as unsure as a uniform
choice among ~6.6 tokens (vs 4,096 at initialisation).

### Costs (ground rule 3)

| | Value |
| --- | --- |
| Params | 3,443,136 (2,656,704 non-embedding) |
| Train throughput | 33.1k–33.7k tokens/s |
| Wall-clock per 50M-token run | ~27 min |
| Peak GPU memory (MPS driver) | ~2.3 GB |
| KV cache, fp32 | 2.25 MB at T = 256 · 36 MB at T = 4,096 |

KV cache = 2 (K and V) × 6 layers × T × 6 KV heads × 32 dims × 4 bytes.
Every layer is global, so it grows linearly with T. This is the number
Experiment 1 tries to shrink.

### Observations

- **Still improving at 50M tokens.** Val loss fell ~0.02 over the last ~600
  steps (seed 1337: 1.9035 at step 5,500 → 1.8830 at 6,103). The model is
  undertrained; comparisons stay fair because every variant gets the same
  50M-token budget.
- **More data matters a lot here.** At LR 6e-3: 10M tokens → 2.246; at 5e-3:
  50M tokens → ~1.88.
- **Train loss ends ~0.025 below val, without overfitting.** Each run sees
  50M of the 100M training tokens, so nearly every training window is new
  text and the train loss is effectively held-out loss on the train split.
  The gap reflects the validation split being slightly harder.
- **Samples are coherent short stories** with consistent characters and simple
  plots (see `runs/control_seed*/sample.txt`). All three seeds open with the
  same sentence because samples use a fixed sampling seed; they diverge after
  a few lines.

Sample (seed 1338, temperature 0.8, top-k 50):

> Once upon a time, there was a little boy named Tim. Tim loved to play with
> his toy cars. One day, he found a big tree. The tree was very tall and had
> many branches that he wanted to touch. Tim thought the tree looked fun and
> he wanted to touch it. [...]

### Caveats

- One dataset (TinyStories), one model size, fp32 on one machine.
- The LR was tuned at 10M tokens and used at 50M.
- Three seeds give a rough noise estimate, not a confidence interval.

---

## Step 2: Gemma 3 270M parity

_Not started._

## Experiment 1: local/global attention

_Not started._
