"""Run configuration: one YAML file fully describes one run.

Why dataclasses instead of a plain dict: a typo like `learning_rate:` instead
of `lr:` fails immediately instead of silently using the default, and every
field has a documented default in one place.
"""

from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class ModelConfig:
    # Names match LlamaModel's arguments, so the model is built with LlamaModel(**model_cfg).
    vocab_size: int = 4096
    d_model: int = 192
    n_heads: int = 6
    d_ff: int = 512
    n_layers: int = 6
    max_seq_len: int = 256  # training context length
    n_kv_heads: int | None = None  # None = same as n_heads (plain multi-head attention)
    head_dim: int | None = None  # None = d_model // n_heads
    tie_embeddings: bool = True
    # Gemma-style options (defaults = the control model, unchanged).
    norm_eps: float = 1e-6
    norm_offset: bool = False  # RMSNorm scale is (1 + w) instead of w
    embed_scale: bool = False  # multiply embeddings by sqrt(d_model)
    qk_norm: bool = False  # RMSNorm on each query/key head before RoPE
    post_norm: bool = False  # extra RMSNorm on each sublayer's output ("sandwich" norm)


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 32
    total_tokens: int = 50_000_000  # token budget; steps = total_tokens / (batch_size * max_seq_len)

    # AdamW + schedule
    lr: float = 3e-3  # peak learning rate, reached at the end of warmup
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Evaluation and printing
    eval_every: int = 250  # steps
    eval_batches: int = 60  # fixed val slice = eval_batches * batch_size * max_seq_len tokens
    log_every: int = 50

    device: str = "auto"  # auto -> cuda, then mps, then cpu
    train_path: str = "data/train.bin"
    val_path: str = "data/val.bin"
    tokenizer_path: str = "data/tokenizer.json"  # used to write sample.txt
    sample_prompt: str = "Once upon a time"


@dataclass
class Config:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def tokens_per_step(self):
        return self.train.batch_size * self.model.max_seq_len

    @property
    def total_steps(self):
        return self.train.total_tokens // self.tokens_per_step

    def to_dict(self):
        return asdict(self)


def _build(cls, raw, section):
    raw = raw or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown key(s) in '{section}': {sorted(unknown)}")
    values = {}
    for key, value in raw.items():
        # YAML gotcha: PyYAML reads `3e-3` as a *string* (it wants `3.0e-3`).
        # Convert so either spelling works.
        if known[key].type is float and isinstance(value, (str, int)):
            value = float(value)
        values[key] = value
    return cls(**values)


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"name", "model", "train"}
    if unknown:
        raise ValueError(f"unknown top-level key(s) in {path}: {sorted(unknown)}")
    return Config(
        name=raw.get("name", "run"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        train=_build(TrainConfig, raw.get("train"), "train"),
    )
