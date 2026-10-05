import torch
import torch.nn as nn

from tiny_frontier.model.norms import RMSNorm
from tiny_frontier.model.rope import RotaryEmbedding
from tiny_frontier.model.block import DecoderBlock


class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        max_seq_len,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, self.rope) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        kv_cache=None,
    ):
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, kv_cache=kv_cache)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits