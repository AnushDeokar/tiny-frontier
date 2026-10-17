import torch
import torch.nn as nn

from tiny_frontier.model.norms import RMSNorm


class KVCache:
    """Keys and values of past tokens, one entry per layer.

    max_tokens (one value per layer, or None) turns a layer's cache into a
    *rolling* cache: after each step only its last `max_tokens` keys/values are
    kept. A local layer with window W never looks further back than W tokens,
    so anything older can be thrown away without changing a single output.
    That is where sliding-window attention actually saves memory.

    Because old entries get dropped, "how many entries are stored" and "how
    many tokens have been processed" are no longer the same number:
      seen[i]     tokens processed so far = absolute position of the next token
                  (RoPE and the mask need this)
      stored(i)   entries currently held (what costs memory)
    """

    def __init__(self, n_layers, max_tokens=None):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.seen = [0] * n_layers
        self.max_tokens = list(max_tokens) if max_tokens is not None else [None] * n_layers

    def length(self, layer_idx):
        """Tokens processed so far (absolute position of the next token)."""
        return self.seen[layer_idx]

    def stored(self, layer_idx):
        k = self.k[layer_idx]
        return 0 if k is None else k.shape[-2]

    def update(self, layer_idx, k, v):
        """Add the new tokens' k, v. Returns (all keys, all values, position of key 0).

        Attention for this step uses everything returned, *then* the cache is
        trimmed. Trimming first would be wrong when several new tokens arrive
        at once (a prompt): the earliest of them still needs keys from before.
        """
        old_k, old_v = self.k[layer_idx], self.v[layer_idx]
        # Position of the oldest key still stored = tokens seen - tokens stored.
        k_start = self.seen[layer_idx] - self.stored(layer_idx)
        self.seen[layer_idx] += k.shape[-2]  # T new tokens
        if old_k is not None:
            k = torch.cat([old_k, k], dim=-2)
            v = torch.cat([old_v, v], dim=-2)

        keep = self.max_tokens[layer_idx]
        if keep is not None and k.shape[-2] > keep:
            self.k[layer_idx], self.v[layer_idx] = k[..., -keep:, :], v[..., -keep:, :]
        else:
            self.k[layer_idx], self.v[layer_idx] = k, v
        return k, v, k_start

    def nbytes(self):
        """Memory held by all stored keys and values right now."""
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)

    def reset(self):
        n = len(self.k)
        self.k, self.v, self.seen = [None] * n, [None] * n, [0] * n


class StreamCache:
    """KV cache for streaming past the training length (Experiment 2).

    Three policies, set by n_sink and window:
      full          n_sink=0, window=None   keep everything
      rolling(W)    n_sink=0, window=W      keep the last W tokens
      sink(k, W)    n_sink=k, window=W      keep the first k tokens forever + the last W
                                            (StreamingLLM)

    Two differences from KVCache:
      * keys are stored BEFORE RoPE, and attention gives each key the position
        of its slot in the cache (0, 1, 2, ...) instead of its place in the
        text. With sink(4, 64) at text position 4,000 the sink tokens sit at
        slots 0-3 and the query at slot 68: every distance stays below the
        255 the model saw in training. Absolute positions would put the sinks
        ~4,000 away, a distance RoPE was never trained on.
      * the first n_sink tokens are never evicted.

    Meant for feeding one token at a time (as the stream evaluator does).
    """

    slot_positions = True

    def __init__(self, n_layers, n_sink=0, window=None):
        if n_sink and window is None:
            raise ValueError("a sink cache needs a window")
        self.n_sink, self.window = n_sink, window
        self.k = [None] * n_layers  # [B, H_kv, slots, head_dim], pre-RoPE
        self.v = [None] * n_layers
        self.seen = [0] * n_layers

    def length(self, layer_idx):
        return self.seen[layer_idx]

    def stored(self, layer_idx):
        k = self.k[layer_idx]
        return 0 if k is None else k.shape[-2]

    def update(self, layer_idx, k, v):
        """Append the new tokens; return (keys, values, number of slots) for
        this step's attention, then evict down to sinks + window."""
        self.seen[layer_idx] += k.shape[-2]
        if self.k[layer_idx] is not None:
            k = torch.cat([self.k[layer_idx], k], dim=-2)
            v = torch.cat([self.v[layer_idx], v], dim=-2)
        n_slots = k.shape[-2]

        keep_k, keep_v = k, v
        if self.window is not None and n_slots > self.n_sink + self.window:
            # The first n_sink slots always hold the first tokens of the
            # stream, because they were never evicted.
            keep_k = torch.cat([k[..., : self.n_sink, :], k[..., -self.window :, :]], dim=-2)
            keep_v = torch.cat([v[..., : self.n_sink, :], v[..., -self.window :, :]], dim=-2)
        self.k[layer_idx], self.v[layer_idx] = keep_k, keep_v
        return k, v, n_slots

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


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
        learned_sink=False,
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

        # Learned attention sink (Experiment 2): one learnable logit per head.
        # Starts at 0, i.e. the sink begins as "worth one average token".
        self.learned_sink = learned_sink
        if learned_sink:
            self.sink_logit = nn.Parameter(torch.zeros(num_heads))

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

        if kv_cache is not None and getattr(kv_cache, "slot_positions", False):
            # Streaming caches (StreamCache): keys are stored *before* RoPE,
            # and positions are the keys' slots in the cache, not their place
            # in the text. Every query-key distance then stays inside the
            # range the model saw in training, however long the stream is.
            k, v, n_slots = kv_cache.update(layer_idx, k.transpose(1, 2), v.transpose(1, 2))
            q = self.rope(q, offset=n_slots - T).transpose(1, 2)
            k = self.rope(k.transpose(1, 2), offset=0).transpose(1, 2)
            k_start, past_len = 0, n_slots - T
        else:
            # RoPE is applied to the new tokens only. Cached K already has it.
            past_len = 0 if kv_cache is None else kv_cache.length(layer_idx)
            q = self.rope(q, offset=past_len)
            k = self.rope(k, offset=past_len)

            # [B, T, H, D] -> [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            k_start = 0  # absolute position of the first key we attend over
            if kv_cache is not None:
                k, v, k_start = kv_cache.update(layer_idx, k, v)

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

        # Positions. Keys run from k_start (0 unless a rolling cache dropped
        # old ones) to the newest token; the T queries are the newest T.
        k_len = k.shape[-2]
        k_pos = k_start + torch.arange(k_len, device=x.device)
        q_pos = past_len + torch.arange(T, device=x.device)
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

        if self.learned_sink:
            # gpt-oss style sink: one extra "column" per head with a learned
            # score and no value vector. Softmax runs over the real scores plus
            # the sink; the sink's share is then dropped, so the weights on real
            # tokens sum to *less than* 1. That is how a head says "attend to
            # nothing" without parking attention on some real token.
            sink = self.sink_logit.view(1, -1, 1, 1).expand(B, -1, T, 1).to(scores.dtype)
            attention_weights = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        else:
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