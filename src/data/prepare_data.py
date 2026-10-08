import hashlib
import json

import numpy as np

from data.tinystories import load_tinystories
from data.tokenizer import EOS_TOKEN, load_tokenizer


def tokenize_split(tokenizer, dataset, split, output_path, max_tokens):
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    token_ids = []
    total_tokens = 0

    for example in dataset[split]:
        ids = tokenizer.encode(example["text"]).ids
        ids.append(eos_id)

        remaining = max_tokens - total_tokens
        ids = ids[:remaining]

        token_ids.extend(ids)
        total_tokens += len(ids)

        if total_tokens >= max_tokens:
            break

    token_ids = np.array(token_ids, dtype=np.uint16)

    with open(output_path, "wb") as f:
        token_ids.tofile(f)

    print(f"{output_path}: {len(token_ids):,} tokens")

    return token_ids


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_fingerprint(paths, output_path):
    fingerprint = {path: file_sha256(path) for path in paths}
    with open(output_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
        f.write("\n")
    print(output_path)
    for path, digest in fingerprint.items():
        print(f"  {path} {digest}")
    return fingerprint


def main():
    dataset = load_tinystories()
    tokenizer = load_tokenizer("data/tokenizer.json")

    train_path = "data/train.bin"
    val_path = "data/val.bin"

    tokenize_split(
        tokenizer=tokenizer,
        dataset=dataset,
        split="train",
        output_path=train_path,
        max_tokens=100_000_000,
    )

    tokenize_split(
        tokenizer=tokenizer,
        dataset=dataset,
        split="validation",
        output_path=val_path,
        max_tokens=1_000_000,
    )

    save_fingerprint(
        [train_path, val_path],
        "data/fingerprint.json",
    )


if __name__ == "__main__":
    main()
