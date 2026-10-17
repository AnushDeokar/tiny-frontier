import numpy as np
import torch


def load_tokens(path):
    # memmap: the 200 MB file stays on disk and the OS pages in only the
    # slices we touch, so RAM use stays tiny.
    return np.memmap(path, dtype=np.uint16, mode="r")


def _chunk(tokens, start, block_size, bos_id):
    """One (x, y) pair of length block_size starting at `start`.

    bos_id=None:  x = t[s .. s+B-1],          y = t[s+1 .. s+B]
    bos_id=k:     x = [k] + t[s .. s+B-2],    y = t[s .. s+B-1]
    With a BOS, position 0 always holds the SAME token, as in Llama-style
    training. The model then predicts the chunk's first real token from <s>.
    """
    # uint16 on disk (vocab < 65,536 halves the file size), int64 for nn.Embedding.
    if bos_id is None:
        x = tokens[start : start + block_size].astype(np.int64)
        y = tokens[start + 1 : start + block_size + 1].astype(np.int64)
    else:
        x = np.concatenate([[bos_id], tokens[start : start + block_size - 1]]).astype(np.int64)
        y = tokens[start : start + block_size].astype(np.int64)
    return x, y


def get_batch(tokens, batch_size, block_size, generator=None, bos_id=None):
    """Random training chunks. y is x shifted by one: predict the next token.

    `generator` gives the data its own random stream, so the batch order
    depends only on the seed, not on anything else that draws random numbers.
    `bos_id`: if set, every chunk starts with that token (see _chunk).
    """
    ix = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    pairs = [_chunk(tokens, i, block_size, bos_id) for i in ix.tolist()]
    x = torch.from_numpy(np.stack([a for a, _ in pairs]))
    y = torch.from_numpy(np.stack([b for _, b in pairs]))
    return x, y


def iter_eval_batches(tokens, batch_size, block_size, n_batches, bos_id=None):
    """Fixed, non-overlapping chunks from the start of the split.

    Unlike get_batch this is deterministic: every evaluation, in every run,
    scores exactly the same tokens. That is what makes val losses comparable
    across runs and across experiments.
    """
    needed = n_batches * batch_size * block_size + 1
    if needed > len(tokens):
        raise ValueError(f"eval needs {needed:,} tokens but the split has {len(tokens):,}")
    for b in range(n_batches):
        starts = [(b * batch_size + j) * block_size for j in range(batch_size)]
        pairs = [_chunk(tokens, s, block_size, bos_id) for s in starts]
        yield torch.from_numpy(np.stack([a for a, _ in pairs])), torch.from_numpy(np.stack([y for _, y in pairs]))
