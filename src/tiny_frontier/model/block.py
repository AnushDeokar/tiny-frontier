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
        norm_eps=1e-6,
        norm_offset=False,
        qk_norm=False,
        post_norm=False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(d_model, eps=norm_eps, offset=norm_offset)  # before attention
        self.norm_2 = RMSNorm(d_model, eps=norm_eps, offset=norm_offset)  # before the MLP

        self.post_norm = post_norm
        if post_norm:
            self.post_attn_norm = RMSNorm(d_model, eps=norm_eps, offset=norm_offset)
            self.post_ff_norm = RMSNorm(d_model, eps=norm_eps, offset=norm_offset)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            rope=rope,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            norm_offset=norm_offset,
        )
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, attention_mask=None, kv_cache=None):
        h = self.attn(
            self.norm_1(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )
        if self.post_norm:
            h = self.post_attn_norm(h)
        x = x + h

        h = self.ff(self.norm_2(x))
        if self.post_norm:
            h = self.post_ff_norm(h)
        x = x + h
        return x