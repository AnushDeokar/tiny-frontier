import torch.nn as nn
from tiny_frontier.model.feed_forward import FeedForward
from tiny_frontier.model.norms import RMSNorm
from tiny_frontier.model.attention import CausalSelfAttention
from tiny_frontier.model.rope import RotaryEmbedding

class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        rope,
        layer_idx=0,
        n_kv_heads=None,
        head_dim=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            rope=rope,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, attention_mask=None, kv_cache=None):
        x = x + self.attn(
            self.norm_1(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )
        x = x + self.ff(self.norm_2(x))
        return x