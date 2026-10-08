import torch.nn.functional as F


def lm_loss(logits, targets):
    B, T, V = logits.shape

    return F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
    )