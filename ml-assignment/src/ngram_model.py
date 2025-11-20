# ngram_model.py
# Core trigram modeling only.
# Uses utilities from utils.py — NO cleaning logic here.

from collections import defaultdict
from src.utils import tokenize, pad_tokens, map_unknowns, choose_from_distribution

class TrigramModel:
    def __init__(self):
        """
        Stores:
        - vocab
        - trigram counts: counts[w1][w2][w3]
        - bigram sets for generation start
        """
        self.vocab = set()
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.fitted = False

    def fit(self, text):
        """Train trigram model."""

        # 1. Tokenize & normalize
        tokens = tokenize(text)

        # 2. Update vocabulary
        for t in tokens:
            self.vocab.add(t)

        # 3. Pad tokens with <START>,<END>
        tokens = pad_tokens(tokens)

        # 4. Map unknowns (first pass vocab includes all tokens)
        tokens = map_unknowns(tokens, self.vocab)

        # 5. Build trigram counts
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigram_counts[w1][w2][w3] += 1

        self.fitted = True

    def generate(self, max_length=50):
        """Generate text probabilistically."""

        if not self.fitted:
            return ""

        # Start with <START>, <START>
        w1, w2 = "<START>", "<START>"
        output = []

        for _ in range(max_length):
            next_candidates = self.trigram_counts[w1][w2]

            if not next_candidates:
                break

            w3 = choose_from_distribution(next_candidates)

            if w3 == "<END>":
                break

            output.append(w3)

            w1, w2 = w2, w3

        return " ".join(output)
