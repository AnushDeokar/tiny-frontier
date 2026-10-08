import numpy as np
import torch


def load_tokens(path):
    # memmap: the 200 MB file stays on disk and the OS pages in only the
    # slices we touch, so RAM use stays tiny.
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(tokens, batch_size, block_size, generator=None):
    """Random training windows. y is x shifted by one: predict the next token.

    `generator` gives the data its own random stream, so the batch order
    depends only on the seed, not on anything else that draws random numbers.
    """
    ix = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    # uint16 on disk (vocab < 65,536 halves the file size), int64 for nn.Embedding.
    x = torch.stack([torch.from_numpy(tokens[i : i + block_size].astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(tokens[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix.tolist()])
    return x, y


def iter_eval_batches(tokens, batch_size, block_size, n_batches):
    """Fixed, non-overlapping windows from the start of the split.

    Unlike get_batch this is deterministic: every evaluation, in every run,
    scores exactly the same tokens. That is what makes val losses comparable
    across runs and across experiments.
    """
    needed = n_batches * batch_size * block_size + 1
    if needed > len(tokens):
        raise ValueError(f"eval needs {needed:,} tokens but the split has {len(tokens):,}")
    for b in range(n_batches):
        starts = [(b * batch_size + j) * block_size for j in range(batch_size)]
        x = np.stack([tokens[s : s + block_size] for s in starts]).astype(np.int64)
        y = np.stack([tokens[s + 1 : s + block_size + 1] for s in starts]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)
