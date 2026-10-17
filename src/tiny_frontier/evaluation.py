"""Evaluation helpers beyond the single mean val loss."""

import numpy as np
import torch
import torch.nn.functional as F

from tiny_frontier.data.loader import iter_eval_batches


@torch.no_grad()
def loss_by_position(model, tokens, batch_size, block_size, n_batches, device, bos_id=None):
    """Mean loss at each position 0..block_size-1 of the evaluation chunks.

    Position p means "the model had p earlier tokens of context" when it
    predicted this token. A local layer with window W sees exactly the same
    tokens as a global layer for p < W, so any penalty from the window can
    only appear at positions p >= W. That makes this curve show *where* the
    window hurts, not just *whether*.
    """
    was_training = model.training
    model.eval()
    total = torch.zeros(block_size, dtype=torch.float64)
    count = 0
    for x, y in iter_eval_batches(tokens, batch_size, block_size, n_batches, bos_id=bos_id):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # reduction="none": keep one loss per token instead of averaging,
        # then add up over the batch, keeping the position axis.
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view(y.shape)
        # Move to CPU *before* converting to float64: Apple's MPS GPU has no
        # float64 support. float64 for the running total keeps rounding error
        # from piling up over thousands of additions.
        total += per_token.sum(dim=0).cpu().double()
        count += y.shape[0]
    model.train(was_training)
    return (total / count).float()


def max_eval_batches(tokens, batch_size, block_size):
    """How many non-overlapping eval batches fit in a token split."""
    return (len(tokens) - 1) // (batch_size * block_size)


@torch.no_grad()
def stream_losses(model, streams, cache, device):
    """Feed long token streams through the model ONE token at a time.

    streams: LongTensor [B, L]. Returns a CPU tensor [B, L-1]: entry [b, t] is
    the loss for predicting token t+1 of stream b after seeing tokens 0..t,
    where what the model can "see" is decided by `cache` (full, rolling,
    sink...). Teacher forcing, as in normal evaluation: the input at every
    step is the real next token, never the model's own guess.
    """
    was_training = model.training
    model.eval()
    x = streams.to(device)
    B, L = x.shape
    losses = torch.empty(B, L - 1, device=device)  # stays on the GPU: no sync per step
    for t in range(L - 1):
        logits = model(x[:, t : t + 1], kv_cache=cache)[:, 0]
        losses[:, t] = F.cross_entropy(logits, x[:, t + 1], reduction="none")
    model.train(was_training)
    return losses.cpu()


def make_streams(tokens, n_streams, length):
    """n_streams non-overlapping slices of `length` tokens from the start of a split."""
    if n_streams * length > len(tokens):
        raise ValueError(f"need {n_streams * length:,} tokens, split has {len(tokens):,}")
    rows = [tokens[i * length : (i + 1) * length].astype("int64") for i in range(n_streams)]
    return torch.from_numpy(np.stack(rows))


@torch.no_grad()
def first_token_attention(model, tokens, batch_size, block_size, n_batches, device, min_query=16, bos_id=None):
    """How much attention each head puts on token 0 of the chunk.

    Returns (mass, sink_share), each [n_layers, n_heads]:
      mass        mean attention weight on key 0, over queries at positions
                  >= min_query (early queries see few tokens, so a high weight
                  on token 0 there means little)
      sink_share  for models with a learned sink: mean weight the head gives
                  the sink ("attend to nothing"); None otherwise
    """
    was_training = model.training
    model.eval()
    attns = [b.attn for b in model.blocks]
    for a in attns:
        a.capture = True
    L, H = len(attns), attns[0].num_heads
    mass = torch.zeros(L, H, dtype=torch.float64)
    sink = torch.zeros(L, H, dtype=torch.float64)
    n = 0
    try:
        for x, _ in iter_eval_batches(tokens, batch_size, block_size, n_batches, bos_id=bos_id):
            model(x.to(device))
            for i, a in enumerate(attns):
                w = a.last_weights[:, :, min_query:, :]  # [B, H, queries, keys]
                # .cpu() BEFORE .double(): the Mac GPU (MPS) has no float64.
                mass[i] += w[..., 0].mean(dim=(0, 2)).cpu().double()
                sink[i] += (1 - w.sum(-1)).mean(dim=(0, 2)).cpu().double()
            n += 1
    finally:
        for a in attns:
            a.capture, a.last_weights = False, None
        model.train(was_training)
    has_sink = attns[0].learned_sink
    return (mass / n).float(), ((sink / n).float() if has_sink else None)
