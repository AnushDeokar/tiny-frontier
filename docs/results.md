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

**Tests.** 56 tests in total. The 29 from Step 1: RoPE (norm preservation, relative-position property),
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

**Goal:** prove the implementation is correct by loading Google's real
Gemma 3 270M weights into *this* codebase and matching Hugging Face
`transformers` output. Inference only, fp32, CPU.

### How Gemma differs from the control

Gemma 3 is the same decoder with every difference added as a config flag
(defaults = the control, guarded by an exact parameter-count test and a
regression test that re-evaluates the trained control checkpoint).
Diagram: [`figures/gemma3_architecture.svg`](figures/gemma3_architecture.svg).

| Difference | Control | Gemma 3 270M | Flag |
| --- | --- | --- | --- |
| RMSNorm scale | `x̂ · w` (w = 1 at init) | `x̂ · (1 + w)` (w = 0 at init) | `norm_offset` |
| Embedding scale | none | × √640 | `embed_scale` |
| QK-norm | none | RMSNorm on each q/k head, before RoPE | `qk_norm` |
| Norms per block | 2 (pre-norm) | 4 (pre + post "sandwich") | `post_norm` |
| MLP | SwiGLU | GeGLU (GELU, tanh approximation) | `mlp_activation` |
| Attention heads | 6 Q / 6 KV, head_dim 32 = d/heads | 4 Q / 1 KV (MQA), head_dim 256 ≠ 640/4 | `n_kv_heads`, `head_dim` |
| Layer pattern | all global | 5 local : 1 global (18 layers) | `layer_pattern` |
| Sliding window | — | 512 tokens | `window` |
| RoPE base | 10,000 | 10,000 local · 1,000,000 global | `rope_base`, `rope_base_local` |

Attention works in 4 × 256 = 1,024 dimensions while the residual stream is
640 wide: `Wq` maps 640 → 1,024 and `Wo` maps back.

63% of "270M" is the embedding table (262,144 × 640 = 167.8M); the 18
transformer layers hold ~100M parameters.

### Method

1. **Read the reference first.** Every difference was taken from the
   installed `transformers` source (`modeling_gemma3.py`, `masking_utils.py`)
   and the checkpoint's `config.json`, not from memory.
2. **Shape check.** Building the model in Gemma's shape gives exactly
   268,098,176 parameters, Google's published count.
3. **Each feature vs Hugging Face's own module**, on random weights
   (RMSNorm, scaled embedding, MLP), plus property tests (QK-norm makes
   attention invariant to the scale of Wq/Wk; window mask sees exactly the
   last W tokens; receptive field grows by W−1 per local layer).
4. **Whole-model parity on a tiny random Gemma** built by `transformers`
   (6 layers, window 8, attention wider than d_model, every weight including
   norms randomised, unusual RoPE bases), logits and greedy decoding.
5. **Real weights.** Converter = renaming only (both sides use
   `nn.Linear` [out, in]; no transposes). All 236 tensors must be used and
   every parameter filled (strict loading both ways). bf16 → fp32.

### Result

| Prompt | Tokens | Max \|logit diff\| | Top-1 agreement | Greedy, 30 tokens |
| --- | --- | --- | --- | --- |
| The capital of France is | 6 | 4.10e-05 | 100% | identical |
| Once upon a time, a little dragon | 9 | 4.96e-05 | 100% | identical |
| def fibonacci(n): | 7 | 6.91e-05 | 100% | identical |
| Water boils at a temperature of | 7 | 6.39e-05 | 100% | identical |
| The three primary colors are | 6 | 5.82e-05 | 100% | identical |
| Model card, first 800 tokens | 800 | 1.81e-04 | 100% | — |

Source: `results/gemma_parity.csv` (`scripts/gemma_parity.py`).

- **Logits agree to ~1e-4** (target was ~1e-3) across all 262,144 vocabulary
  entries at every position. The remaining difference is fp32 rounding from
  operations done in a different order over 18 layers.
- **The 800-token prompt is the sliding-window test.** The five short
  prompts never exceed the 512-token window, so local layers never cut
  anything; the long one does, and still agrees on the top-1 token at all
  800 positions. Its slightly larger difference (1.8e-4) is rounding
  accumulating over more positions.
- **Greedy generation is identical** (our KV-cached decoding vs Hugging
  Face), and the text is sensible: "The capital of France is Paris…",
  a correct `fibonacci` function, water boils at 100 °C.
- The generations repeat themselves ("It is the most visited city in the
  world. It is the most visited city…"). That is expected from a small
  *base* (not instruction-tuned) model under greedy decoding, not a bug:
  it matches Hugging Face token for token.

### Pitfalls (the traps this checks for)

Parity matched on the first run on real weights, because every feature had
already been matched against Hugging Face's modules on random weights. The
traps that approach caught or avoided:

- **`(1 + w)` norms.** Gemma stores norm weights near 0. Read with the Llama
  convention they would scale every activation by ~0, silently.
- **The `post_attention_layernorm` name.** In HF Llama it is the norm
  *before* the MLP; in Gemma 3 it is the norm on the attention *output*.
  Same string, different job; mapped by function, not by name.
- **`head_dim` ≠ d_model / heads.** Projections must be sized by
  heads × head_dim (1,024), or the weights don't even load.
- **GELU variant.** Gemma uses the tanh approximation; exact GELU differs by
  up to ~1e-3, enough to break parity.
- **No `lm_head` in the checkpoint.** The output layer is tied to the
  embedding; Google stores the 168M-parameter matrix once.
- **bf16 checkpoint.** Compared in fp32; bf16's ~3 significant digits would
  hide real bugs.
- **Short prompts don't test the window.** Needed an input longer than 512
  tokens.
- **Random-weight tests must randomise norms too.** A zero-initialised Gemma
  norm makes `w` and `1 + w` look identical for a Llama norm at init.

### Not covered

- The local layers' KV cache still stores every token (correct, via the
  mask, but no memory saved yet); a rolling cache comes in Experiments 1-2.
- Checked on CPU in fp32 only, and only for the 270M model.

## Experiment 1: local/global attention

_Not started._
