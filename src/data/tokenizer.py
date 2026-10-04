from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def create_bpe_tokenizer():
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer

def train_tokenizer(tokenizer, dataset, vocab_size=4096):
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "</s>", "<pad>"],
    )
    tokenizer.train_from_iterator(
        (row["text"] for row in dataset["train"]),
        trainer=trainer,
    )
    return tokenizer

def save_tokenizer(tokenizer, path):
    tokenizer.save(path)
    return tokenizer

def load_tokenizer(path):
    tokenizer = Tokenizer.from_file(path)
    return tokenizer

def tokenize_text(tokenizer, text):
    return tokenizer.encode(text).ids

def decode_text(tokenizer, ids):
    return tokenizer.decode(ids)