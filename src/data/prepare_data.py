import numpy as np
from data.tinystories import load_tinystories
from data.tokenizer import load_tokenizer

def tokenize_split(tokenizer, dataset, split, output_path, max_tokens):
    token_ids = []
    total_tokens = 0

    for example in dataset[split]:
        ids = tokenizer.encode(example["text"]).ids

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

def main():
    dataset = load_tinystories()
    tokenizer = load_tokenizer("data/tokenizer.json")

    tokenize_split(
        tokenizer=tokenizer,
        dataset=dataset,
        split="train",
        output_path="data/train.bin",
        max_tokens=100_000_000,
    )

    tokenize_split(
        tokenizer=tokenizer,
        dataset=dataset,
        split="validation",
        output_path="data/val.bin",
        max_tokens=1_000_000,
    )


if __name__ == "__main__":
    main()