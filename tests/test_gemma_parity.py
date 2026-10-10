"""Parity with Hugging Face's Gemma 3: same weights in, same logits out.

1. A tiny Gemma 3 with random weights, created by Hugging Face's own code.
   Runs in seconds and needs no download.
2. The real google/gemma-3-270m weights (marked slow; skipped if the
   checkpoint is not downloaded).
"""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Gemma3ForCausalLM, Gemma3TextConfig  # noqa: E402

from tiny_frontier.convert.gemma3 import config_from_hf, convert_state_dict, load_gemma3  # noqa: E402
from tiny_frontier.generate import generate  # noqa: E402
from tiny_frontier.model.model import LlamaModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "gemma-3-270m"


def tiny_hf_gemma():
    cfg = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,  # default pattern: 5 local + 1 global
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,  # 4 x 32 = 128 != 64: attention wider than d_model, like the real one
        query_pre_attn_scalar=32,
        sliding_window=8,
        max_position_embeddings=256,
        # Unusual values on purpose: if we read the wrong key, the test notices.
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 500_000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 20_000.0},
        },
    )
    model = Gemma3ForCausalLM(cfg).eval()
    with torch.no_grad():
        # Randomise everything, norms included, so no weight is right by accident
        # (e.g. a zero-initialised Gemma norm would hide a w vs 1+w mix-up).
        for p in model.parameters():
            p.normal_(0, 0.1)
    return model


def ours_from_hf(hf_model):
    cfg = config_from_hf(hf_model.config.to_dict())
    model = LlamaModel(**asdict(cfg))
    model.load_state_dict(convert_state_dict(hf_model.state_dict(), cfg.n_layers))
    return model.eval(), cfg


@torch.no_grad()
def test_tiny_random_gemma_matches_hugging_face():
    hf = tiny_hf_gemma()
    ours, cfg = ours_from_hf(hf)
    assert cfg.layer_pattern == "LLLLLG"
    assert (cfg.rope_base, cfg.rope_base_local) == (500_000.0, 20_000.0)

    ids = torch.randint(0, 128, (2, 24))  # 24 > window 8, so the local layers really cut
    diff = (ours(ids) - hf(ids).logits).abs().max().item()
    assert diff < 1e-4, f"max logit difference {diff:.2e}"


@torch.no_grad()
def test_tiny_random_gemma_greedy_decoding_matches():
    # Ours: KV cache, one token at a time. Reference: Hugging Face re-running
    # the full sequence each step. Same tokens = cache + windows + RoPE offsets agree.
    hf = tiny_hf_gemma()
    ours, _ = ours_from_hf(hf)
    prompt = torch.randint(0, 128, (1, 5))
    got = generate(ours, prompt, max_new_tokens=15, temperature=0)

    ref = prompt
    for _ in range(15):
        next_id = hf(ref).logits[:, -1].argmax(-1, keepdim=True)
        ref = torch.cat([ref, next_id], dim=1)
    assert torch.equal(got, ref)


def test_unmapped_tensors_are_an_error():
    hf = tiny_hf_gemma()
    sd = dict(hf.state_dict())
    sd["model.layers.0.self_attn.mystery.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="not mapped"):
        convert_state_dict(sd, n_layers=6)


@pytest.mark.slow
@pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="gemma-3-270m not downloaded")
@torch.no_grad()
def test_real_gemma_3_270m_matches_hugging_face():
    from transformers import AutoTokenizer

    ours, _ = load_gemma3(CKPT)
    hf = Gemma3ForCausalLM.from_pretrained(CKPT, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(CKPT)

    ids = tok("The capital of France is", return_tensors="pt").input_ids
    diff = (ours(ids) - hf(ids).logits).abs().max().item()
    assert diff < 1e-3, f"max logit difference {diff:.2e}"

    got = generate(ours, ids, max_new_tokens=10, temperature=0)
    ref = hf.generate(ids, max_new_tokens=10, do_sample=False)
    n = min(got.shape[1], ref.shape[1])
    assert torch.equal(got[:, :n], ref[:, :n])
