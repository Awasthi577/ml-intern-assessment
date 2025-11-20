import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.utils import tokenize, clean_token, split_punctuation, pad_tokens, map_unknowns, choose_from_distribution
from src.ngram_model import TrigramModel
 
def test_clean_token_basic():
    assert clean_token("Hello") == "hello"
    assert clean_token("Heeellp") == "help"
    assert clean_token("WORLD!!!") == "world"
    assert clean_token("123$$") == "123"
    assert clean_token("") == ""


def test_split_punctuation():
    assert split_punctuation("Hello, world!") == ["Hello", ",", "world", "!"]
    assert split_punctuation("A.B?C") == ["A", ".", "B", "?", "C"]
    assert split_punctuation("") == []


def test_tokenize_full():
    assert tokenize("Hello,   WORLD!!!") == ["hello", "world"]
    assert tokenize("Hi? Bye!") == ["hi", "?", "bye", "!"]
    assert tokenize("Heeellloooo...??") == ["helo", ".", ".", ".", "?", "?"]


def test_pad_tokens():
    assert pad_tokens(["a", "b"]) == ["<START>", "<START>", "a", "b", "<END>"]
    assert pad_tokens([]) == ["<START>", "<START>", "<END>"]


def test_map_unknowns_basic():
    vocab = {"hello", "world"}
    tokens = ["hello", "mars", "world"]
    assert map_unknowns(tokens, vocab) == ["hello", "<UNK>", "world"]


def test_choose_from_distribution():
    dist = {"a": 1, "b": 1, "c": 1}
    for _ in range(20):
        assert choose_from_distribution(dist) in {"a", "b", "c"}


def test_trigram_fit_counts():
    m = TrigramModel()
    m.fit("I am good.")
    assert m.trigram_counts["<START>"]["<START>"]["i"] == 1
    assert m.trigram_counts["i"]["am"]["good"] == 1


def test_generate_basic():
    m = TrigramModel()
    m.fit("I am a test.")
    out = m.generate()
    assert isinstance(out, str)


def test_generate_empty_input():
    m = TrigramModel()
    m.fit("")
    assert m.generate() == ""


def test_generate_short_input():
    m = TrigramModel()
    m.fit("Hello.")
    assert isinstance(m.generate(), str)


def test_unicode_handling():
    m = TrigramModel()
    m.fit("Café naïve façade. Hello!")
    out = m.generate()
    assert isinstance(out, str)


def test_numeric_tokens():
    m = TrigramModel()
    m.fit("I have 2 dogs and 3 cats.")
    out = m.generate()
    assert isinstance(out, str)


def test_weird_symbols():
    m = TrigramModel()
    m.fit("Hello @#$ world %^& test.")
    out = m.generate()
    assert isinstance(out, str)


def test_repeated_letter_text():
    m = TrigramModel()
    m.fit("Heeellooo wooorld.")
    out = m.generate()
    assert isinstance(out, str)


def test_no_crash_large_repetition():
    m = TrigramModel()
    m.fit("ha" * 500)
    out = m.generate()
    assert isinstance(out, str)


def test_model_builds_vocab():
    m = TrigramModel()
    m.fit("I love pizza.")
    assert "i" in m.vocab
    assert "love" in m.vocab


def test_unknown_mapping():
    m = TrigramModel()
    m.fit("hello world")
    tokens = map_unknowns(["hello", "mars"], m.vocab)
    assert tokens == ["hello", "<UNK>"]


def test_padding_presence():
    m = TrigramModel()
    m.fit("hello world")
    assert "<START>" in m.trigram_counts
    assert "hello" in m.trigram_counts["<START>"]


def test_sampling_distribution_edge_case():
    assert choose_from_distribution({"a": 10}) == "a"


def test_generate_sentence_max_length():
    m = TrigramModel()
    m.fit("I am testing this model. It should generate correctly.")
    out = m.generate(max_length=5)
    assert len(out.split()) <= 5


def test_fit_multiple_sentences():
    m = TrigramModel()
    m.fit("I am good. You are great. We are awesome.")
    out = m.generate()
    assert isinstance(out, str)


def test_symbol_heavy_string():
    m = TrigramModel()
    m.fit("!!!! ??? .... ,, ,,, !!!!")
    out = m.generate()
    assert isinstance(out, str)


def test_context_switching():
    m = TrigramModel()
    m.fit("I love coding. I love pizza. I love music.")
    out = m.generate()
    assert "love" in out or out == ""
