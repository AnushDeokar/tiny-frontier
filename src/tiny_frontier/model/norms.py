import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm: x / sqrt(mean(x^2) + eps), then a learned per-channel scale.

    Two conventions for that scale exist in real models:
      offset=False (Llama):  y = x_norm * w,        w starts at 1
      offset=True  (Gemma):  y = x_norm * (1 + w),  w starts at 0
    Both start as the identity scale; they differ in what number is *stored*.
    Loading Gemma's weights (w near 0) into the Llama convention would
    multiply every activation by roughly zero, silently.
    """

    def __init__(self, d_model, eps=1e-6, offset=False):
        super().__init__()
        self.eps = eps
        self.offset = offset
        init = torch.zeros(d_model) if offset else torch.ones(d_model)
        self.weight = nn.Parameter(init)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        square_mean = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(square_mean + self.eps)
        scale = (1.0 + self.weight.float()) if self.offset else self.weight
        return (scale * x).to(input_dtype)
