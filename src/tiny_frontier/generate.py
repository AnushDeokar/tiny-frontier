import torch

from tiny_frontier.model.attention import KVCache


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    eos_id=None,
    use_cache=True,
    generator=None,
):
    """Autoregressive generation: predict a token, append it, repeat.

    With use_cache=True the prompt is processed once, and after that each
    step feeds only the single new token; past keys/values come from the
    KV cache. Without the cache, every step re-runs the whole sequence,
    which is O(T^2) work instead of O(T).

    temperature=0 means greedy (always the most likely token).
    top_k keeps only the k most likely tokens before sampling, which cuts
    off the long tail of nonsense tokens a small model assigns tiny
    probabilities to.
    """
    was_training = model.training
    model.eval()

    cache = KVCache(len(model.blocks)) if use_cache else None
    ids = input_ids
    next_input = input_ids  # first step: the whole prompt

    for _ in range(max_new_tokens):
        if use_cache:
            logits = model(next_input, kv_cache=cache)
        else:
            logits = model(ids)
        logits = logits[:, -1, :]  # only the last position predicts the next token

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            # Sample on CPU so a CPU torch.Generator makes it reproducible on any device.
            next_id = torch.multinomial(probs.cpu(), 1, generator=generator).to(ids.device)

        ids = torch.cat([ids, next_id], dim=1)
        next_input = next_id
        if eos_id is not None and (next_id == eos_id).all():
            break

    model.train(was_training)
    return ids
