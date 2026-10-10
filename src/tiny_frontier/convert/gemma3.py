"""Load Hugging Face Gemma 3 weights into our model.

Two pieces:
  config_from_hf      Gemma's config.json  -> our ModelConfig (flags + sizes)
  convert_state_dict  Gemma's tensor names -> our tensor names

No tensor is reshaped or transposed: both codebases use nn.Linear, which
stores weights as [out_features, in_features]. Only the names change.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_frontier.config import ModelConfig
from tiny_frontier.model.model import LlamaModel

# Per-layer names: Hugging Face (after "layers.{i}.")  ->  ours (after "blocks.{i}.")
LAYER_MAP = {
    "input_layernorm.weight": "norm_1.weight",  # before attention
    # Naming trap: in Hugging Face *Llama*, "post_attention_layernorm" is the
    # norm before the MLP. In *Gemma 3* it is the norm applied to the attention
    # output (a sandwich norm). We map by what it does, not by its name.
    "post_attention_layernorm.weight": "post_attn_norm.weight",
    "pre_feedforward_layernorm.weight": "norm_2.weight",  # before the MLP
    "post_feedforward_layernorm.weight": "post_ff_norm.weight",
    "self_attn.q_proj.weight": "attn.Wq.weight",
    "self_attn.k_proj.weight": "attn.Wk.weight",
    "self_attn.v_proj.weight": "attn.Wv.weight",
    "self_attn.o_proj.weight": "attn.Wo.weight",
    "self_attn.q_norm.weight": "attn.q_norm.weight",
    "self_attn.k_norm.weight": "attn.k_norm.weight",
    "mlp.gate_proj.weight": "ff.W_gate.weight",
    "mlp.up_proj.weight": "ff.W_up.weight",
    "mlp.down_proj.weight": "ff.W_down.weight",
}

ACTIVATIONS = {"gelu_pytorch_tanh": "gelu_tanh", "silu": "silu"}


def config_from_hf(hf, max_seq_len=1024):
    """Translate a Gemma 3 text config (dict from config.json) into ModelConfig.

    Fails loudly on any Gemma option this codebase does not implement, so a
    parity mismatch can't be caused by a silently ignored setting.
    """
    head_dim = hf["head_dim"]
    unsupported = {
        "query_pre_attn_scalar": hf.get("query_pre_attn_scalar", head_dim) != head_dim,
        "attn_logit_softcapping": hf.get("attn_logit_softcapping") is not None,
        "final_logit_softcapping": hf.get("final_logit_softcapping") is not None,
        "attention_bias": bool(hf.get("attention_bias", False)),
        "tie_word_embeddings": not hf.get("tie_word_embeddings", True),
        "hidden_activation": hf["hidden_activation"] not in ACTIVATIONS,
    }
    bad = [name for name, is_bad in unsupported.items() if is_bad]
    if bad:
        raise ValueError(f"unsupported Gemma settings: {bad}")

    # RoPE bases: older configs have flat keys, newer ones a per-layer-type dict.
    rope = hf.get("rope_parameters") or {}
    if "full_attention" in rope:
        rope_global = rope["full_attention"]["rope_theta"]
        rope_local = rope["sliding_attention"]["rope_theta"]
    else:
        rope_global = hf.get("rope_theta", 1_000_000.0)
        rope_local = hf.get("rope_local_base_freq", 10_000.0)

    # ["sliding_attention", ..., "full_attention", ...] -> "LLLLLG..."
    pattern = "".join("L" if t == "sliding_attention" else "G" for t in hf["layer_types"])

    return ModelConfig(
        vocab_size=hf["vocab_size"],
        d_model=hf["hidden_size"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf["num_key_value_heads"],
        head_dim=head_dim,
        d_ff=hf["intermediate_size"],
        n_layers=hf["num_hidden_layers"],
        max_seq_len=max_seq_len,
        tie_embeddings=True,
        norm_eps=hf["rms_norm_eps"],
        norm_offset=True,
        embed_scale=True,
        qk_norm=True,
        post_norm=True,
        mlp_activation=ACTIVATIONS[hf["hidden_activation"]],
        layer_pattern=pattern,
        window=hf["sliding_window"],
        rope_base=float(rope_global),
        rope_base_local=float(rope_local),
    )


def convert_state_dict(hf_sd, n_layers):
    """Rename Gemma tensors to ours. Every input tensor must be used exactly once."""
    # The prefix varies ("model.", or "model.language_model." in multimodal
    # checkpoints), so find it from the embedding's name.
    embed_key = next(k for k in hf_sd if k.endswith("embed_tokens.weight"))
    prefix = embed_key[: -len("embed_tokens.weight")]

    ours, used = {}, set()

    def take(hf_name, our_name):
        ours[our_name] = hf_sd[hf_name]
        used.add(hf_name)

    take(embed_key, "token_embedding.weight")
    take(prefix + "norm.weight", "norm.weight")
    for i in range(n_layers):
        for hf_suffix, our_suffix in LAYER_MAP.items():
            take(f"{prefix}layers.{i}.{hf_suffix}", f"blocks.{i}.{our_suffix}")

    # Tied output layer: same tensor as the embedding. Our state_dict lists it
    # under both names, so provide both. A stored lm_head (if any) is skipped.
    ours["lm_head.weight"] = ours["token_embedding.weight"]
    used.update(k for k in hf_sd if k.endswith("lm_head.weight"))

    leftover = sorted(set(hf_sd) - used)
    if leftover:
        raise ValueError(f"{len(leftover)} Gemma tensors were not mapped, e.g. {leftover[:3]}")
    return ours


def load_gemma3(checkpoint_dir, dtype=torch.float32, max_seq_len=1024):
    """Build our model from a downloaded Hugging Face Gemma 3 checkpoint."""
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    hf_cfg = json.loads((checkpoint_dir / "config.json").read_text())
    cfg = config_from_hf(hf_cfg, max_seq_len=max_seq_len)

    hf_sd = load_file(checkpoint_dir / "model.safetensors")
    # The checkpoint is bf16; parity is checked in fp32 so rounding stays tiny.
    hf_sd = {k: v.to(dtype) for k, v in hf_sd.items()}

    model = LlamaModel(**asdict(cfg))
    # strict=True: every one of our parameters must be filled, nothing extra.
    model.load_state_dict(convert_state_dict(hf_sd, cfg.n_layers), strict=True)
    return model.to(dtype).eval(), cfg
