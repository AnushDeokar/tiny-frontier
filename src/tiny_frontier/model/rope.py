import torch
import torch.nn as nn


def _rotate_half(x):
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"sequence length {end} exceeds max_seq_len {self.max_seq_len}"
            )
        middle = [1] * (x.ndim - 3)
        cos = self.cos[offset:end].to(dtype=x.dtype).view(1, seq_len, *middle, -1)
        sin = self.sin[offset:end].to(dtype=x.dtype).view(1, seq_len, *middle, -1)
        return x * cos + _rotate_half(x) * sin
