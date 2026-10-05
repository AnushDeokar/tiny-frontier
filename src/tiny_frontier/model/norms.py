import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        square_mean = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(square_mean + self.eps)
        return (self.weight * x).to(input_dtype)