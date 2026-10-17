from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<s>", "</s>", "<pad>"]
EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
BOS_ID = SPECIAL_TOKENS.index(BOS_TOKEN)  # 0: special tokens get the first ids


def create_bpe_tokenizer():
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    return tokenizer


def train_tokenizer(tokenizer, dataset, vocab_size=4096):
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
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
