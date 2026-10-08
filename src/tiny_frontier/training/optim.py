import math

import torch


def build_optimizer(model, cfg):
    """AdamW with two parameter groups.

    Weight decay pulls weights toward zero. That is a useful regulariser for
    the big matrices (attention, MLP, embeddings), but for RMSNorm gains it
    just fights the norm's job of setting a scale, so 1-D params get none.
    Rule of thumb from nanoGPT: decay if p.dim() >= 2.
    """
    decay, no_decay = [], []
    # named_parameters() yields the tied embedding/lm_head weight only once.
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def lr_at(step, total_steps, cfg):
    """Linear warmup, then cosine decay to lr * min_lr_ratio.

    Warmup: at step 0 Adam's moment estimates are empty, so its first updates
    are noisy; a small LR stops them from wrecking the fresh init.
    Cosine: big steps early to make progress, small steps late to settle into
    a good minimum.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    min_lr = cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - min_lr)
