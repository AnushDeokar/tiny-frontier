"""RoPE rotates pairs of dimensions by an angle that depends on position.

Two properties follow, and if either fails the implementation is wrong:
1. A rotation never changes a vector's length.
2. The score q.k depends only on the distance between positions,
   not on where they are. That is the whole point of "relative" position.
"""

import torch

from tiny_frontier.model.rope import RotaryEmbedding


def test_rope_preserves_norm():
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    x = torch.randn(2, 10, 4, 32)  # [B, T, H, head_dim]
    out = rope(x)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))


def test_rope_position_zero_is_identity():
    # At position 0 the angle is 0: cos = 1, sin = 0, so nothing changes.
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    x = torch.randn(1, 1, 1, 32)
    torch.testing.assert_close(rope(x, offset=0), x)


def test_rope_score_depends_only_on_offset():
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)

    def score(q_pos, k_pos):
        return (rope(q, offset=q_pos) * rope(k, offset=k_pos)).sum()

    # Same distance (3) at different places -> same score.
    torch.testing.assert_close(score(5, 2), score(25, 22), atol=1e-5, rtol=1e-5)
    # Different distance -> different score (otherwise RoPE would encode nothing).
    assert not torch.allclose(score(5, 2), score(5, 4), atol=1e-4)
