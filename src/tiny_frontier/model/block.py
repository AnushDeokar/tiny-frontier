import torch.nn as nn
from tiny_frontier.model.feed_forward import FeedForward
from tiny_frontier.model.norms import RMSNorm
from tiny_frontier.model.attention import CausalSelfAttention
from tiny_frontier.model.rope import RotaryEmbedding

class DecoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, rope, dropout=0.1):
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope=rope)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, attention_mask=None, kv_cache=None):
        x = x + self.attn(self.norm_1(x), attention_mask=attention_mask, kv_cache=kv_cache)
        x = x + self.ff(self.norm_2(x))
        return x