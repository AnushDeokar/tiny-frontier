import torch.nn as nn
import torch.nn.functional as F

# The gate's activation is the only difference between the two gated MLPs:
#   SwiGLU (Llama, the control): silu(gate) * up
#   GeGLU  (Gemma):              gelu(gate) * up   (tanh approximation of GELU)
ACTIVATIONS = {
    "silu": F.silu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


class FeedForward(nn.Module):
    """Gated MLP: down( act(gate(x)) * up(x) ).

    The gate branch decides, per hidden unit, how much of the up branch
    passes through. Three matrices instead of the classic two, which is why
    gated MLPs use a smaller hidden size (e.g. 512 here instead of 4*192).
    """

    def __init__(self, d_model, d_ff, activation="silu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")
        self.activation = activation
        self.act = ACTIVATIONS[activation]
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.W_down(self.act(self.W_gate(x)) * self.W_up(x))
