import numpy as np
import torch

def load_tokens(path):
    return np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

def get_batch(tokens, batch_size, block_size):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(np.array(tokens[i:i+block_size])) for i in ix])
    y = torch.stack([torch.from_numpy(np.array(tokens[i+1:i+block_size+1])) for i in ix])
    return x, y
