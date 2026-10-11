"""Evaluation helpers beyond the single mean val loss."""

import torch
import torch.nn.functional as F

from tiny_frontier.data.loader import iter_eval_batches


@torch.no_grad()
def loss_by_position(model, tokens, batch_size, block_size, n_batches, device):
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
    for x, y in iter_eval_batches(tokens, batch_size, block_size, n_batches):
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
