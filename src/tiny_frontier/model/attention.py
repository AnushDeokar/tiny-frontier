import torch
import torch.nn as nn


class KVCache:
    
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def length(self, layer_idx):
        if self.k[layer_idx] is None:
            return 0
        return self.k[layer_idx].shape[-2]

    def update(self, layer_idx, k, v):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=-2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=-2)
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None


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

    def forward(self, x, attention_mask=None, kv_cache=None, layer_idx=0):
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

        # RoPE is applied to the new tokens only. Cached K already has it.
        past_len = 0 if kv_cache is None else kv_cache.length(layer_idx)
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # QK^T / sqrt(d_head)
        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # Query i attends to keys through past_len + i.
        k_len = k.shape[-2]
        causal_mask = torch.triu(
            torch.ones(
                T,
                k_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=k_len - T + 1,
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