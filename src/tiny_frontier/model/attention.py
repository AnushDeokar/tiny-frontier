import torch
import torch.nn as nn

from tiny_frontier.model.norms import RMSNorm


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
    
    def __init__(
        self,
        d_model,
        num_heads,
        rope,
        n_kv_heads=None,
        head_dim=None,
        qk_norm=False,
        norm_eps=1e-6,
        norm_offset=False,
        window=None,
    ):
        super().__init__()

        # None = global attention (sees every earlier token).
        # An int W = local attention (sees only the last W tokens).
        self.window = window


        if n_kv_heads is None:
            n_kv_heads = num_heads
        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by "
                    f"num_heads ({num_heads}) when head_dim is not set"
                )
            head_dim = d_model // num_heads
        if num_heads % n_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"n_kv_heads ({n_kv_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        # Grouped-query attention: each K/V head is shared by n_rep query heads.
        self.n_rep = num_heads // n_kv_heads
        self.head_dim = head_dim
        self.rope = rope

        # head_dim can differ from d_model // num_heads (Gemma 3 does this),
        # so the projections are sized by heads * head_dim, not d_model.
        self.Wq = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.Wo = nn.Linear(num_heads * head_dim, d_model, bias=False)

        # QK-norm (Gemma 3): RMSNorm each query and key head vector (size
        # head_dim) before RoPE. Keeps q.k from growing large, which keeps
        # the softmax from becoming extremely peaked.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps, offset=norm_offset)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps, offset=norm_offset)

    def forward(self, x, attention_mask=None, kv_cache=None, layer_idx=0):
        B, T, C = x.shape

        q = self.Wq(x).view(
            B, T, self.num_heads, self.head_dim
        )

        k = self.Wk(x).view(
            B, T, self.n_kv_heads, self.head_dim
        )

        v = self.Wv(x).view(
            B, T, self.n_kv_heads, self.head_dim
        )

        # [B, T, heads, head_dim]: RMSNorm acts on the last dim, i.e. per head.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

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

        # The cache stores only n_kv_heads (that's the memory saving).
        # Expand to num_heads just for the score computation:
        # [B, H_kv, S, D] -> [B, H_kv * n_rep, S, D]
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # QK^T / sqrt(d_head)
        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # Absolute positions. The cache holds keys 0..k_len-1 and the T new
        # queries are the last T of those positions.
        k_len = k.shape[-2]
        k_pos = torch.arange(k_len, device=x.device)
        q_pos = k_pos[k_len - T :]
        # True = blocked. Causal: a query may not look at later keys.
        blocked = k_pos[None, :] > q_pos[:, None]
        if self.window is not None:
            # Sliding window: query i sees only keys i-W+1 .. i (the last W
            # tokens, itself included). Same rule as Hugging Face's
            # `kv_idx > q_idx - sliding_window`.
            blocked = blocked | (k_pos[None, :] <= q_pos[:, None] - self.window)
        scores = scores.masked_fill(blocked, float("-inf"))

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
        out = out.view(B, T, self.num_heads * self.head_dim)

        # Wo
        out = self.Wo(out)

        return out