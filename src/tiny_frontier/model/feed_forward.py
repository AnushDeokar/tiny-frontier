import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff)
        self.W_up = nn.Linear(d_model, d_ff)
        self.W_down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
