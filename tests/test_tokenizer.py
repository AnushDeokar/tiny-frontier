from pathlib import Path

import pytest

from tiny_frontier.data.tokenizer import EOS_TOKEN, load_tokenizer

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "data" / "tokenizer.json"

pytestmark = pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="tokenizer.json not built")


@pytest.mark.parametrize(
    "text",
    ["Hello, world!", "One day, a little girl named Lily went to the park.", "café ☃  two  spaces"],
)
def test_round_trip(text):
    # Byte-level BPE must give back exactly the input: spaces, punctuation, unicode.
    tok = load_tokenizer(str(TOKENIZER_PATH))
    assert tok.decode(tok.encode(text).ids) == text


def test_vocab_and_eos():
    tok = load_tokenizer(str(TOKENIZER_PATH))
    assert tok.get_vocab_size() == 4096
    assert tok.token_to_id(EOS_TOKEN) == 1
