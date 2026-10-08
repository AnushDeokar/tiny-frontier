import torch

from tiny_frontier.model.norms import RMSNorm


def test_rmsnorm_matches_formula():
    norm = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
    x = torch.randn(3, 5, 16)

    # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight. No mean subtraction,
    # no bias: that is the difference from LayerNorm.
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected)


def test_rmsnorm_matches_torch():
    if not hasattr(torch.nn, "RMSNorm"):  # added in torch 2.4
        return
    ours = RMSNorm(16, eps=1e-6)
    ref = torch.nn.RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        w = torch.randn(16)
        ours.weight.copy_(w)
        ref.weight.copy_(w)
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(ours(x), ref(x))
