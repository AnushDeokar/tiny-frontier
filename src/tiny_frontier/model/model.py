import math

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
        rope_max_len=8192,
        n_kv_heads=None,
        head_dim=None,
        tie_embeddings=True,
        init_std=0.02,
        norm_eps=1e-6,
        norm_offset=False,
        embed_scale=False,
    ):
        super().__init__()

        self.n_layers = n_layers
        # Training context length (what the model sees during training).
        self.max_seq_len = max_seq_len
        self.init_std = init_std
        # Gemma multiplies embeddings by sqrt(d_model). With tied embeddings the
        # same matrix is also the output layer, so its values are kept small;
        # the multiplier brings the *input* side back up to a useful scale.
        self.embed_multiplier = math.sqrt(d_model) if embed_scale else 1.0

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if head_dim is None:
            head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, max(max_seq_len, rope_max_len))
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model,
                n_heads,
                d_ff,
                self.rope,
                layer_idx=i,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                norm_eps=norm_eps,
                norm_offset=norm_offset,
            )
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model, eps=norm_eps, offset=norm_offset)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tied embeddings: the output projection reuses the input embedding matrix.
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # stream get a smaller std, so the residual doesn't grow with depth.
        residual_std = init_std / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.Wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ff.W_down.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def embed(self, input_ids):
        x = self.token_embedding(input_ids)
        if self.embed_multiplier != 1.0:
            x = x * self.embed_multiplier
        return x

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    def forward(
        self,
        input_ids,
        attention_mask=None,
        kv_cache=None,
    ):
        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, kv_cache=kv_cache)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits
