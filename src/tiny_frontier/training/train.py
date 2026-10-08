import torch

from data.loader import load_tokens, get_batch
from tiny_frontier.model.model import LlamaModel
from tiny_frontier.training.loss import lm_loss


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

VOCAB_SIZE = 4096
D_MODEL = 128
N_HEADS = 4
D_FF = 352
N_LAYERS = 4
BLOCK_SIZE = 256

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
STEPS = 5000


def main():
    train_tokens = load_tokens("data/train.bin")
    val_tokens = load_tokens("data/val.bin")

    model = LlamaModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_seq_len=BLOCK_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Device: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(STEPS):
        x, y = get_batch(
            train_tokens,
            BATCH_SIZE,
            BLOCK_SIZE,
        )

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = lm_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}")


if __name__ == "__main__":
    main()