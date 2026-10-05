import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, rope):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attention_mask=None, kv_cache=None):
        B, T, C = x.shape

        q = self.Wq(x).view(
            B, T, self.num_heads, self.head_dim
        )

        k = self.Wk(x).view(
            B, T, self.num_heads, self.head_dim
        )

        v = self.Wv(x).view(
            B, T, self.num_heads, self.head_dim
        )

        # RoPE is applied to Q and K, not V
        q = self.rope(q)
        k = self.rope(k)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK^T / sqrt(d_head)
        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask,
                float("-inf"),
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # Attention(Q,K,V)
        out = attention_weights @ v

        # [B, H, T, D] -> [B, T, H, D]
        out = out.transpose(1, 2).contiguous()

        # Concatenate heads
        out = out.view(B, T, self.d_model)

        # Wo
        out = self.Wo(out)

        return out