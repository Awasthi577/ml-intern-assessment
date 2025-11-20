# This file is optional.
# You can add any utility functions you need for your implementation here.
import re
import random


def clean_token(token):
    if not token:
        return ""

    token = token.lower()
    token = re.sub(r"[.,?!]+$", "", token)

    if re.search(r"(.)\1{2,}", token):
        token = re.sub(r"(.)\1+", r"\1", token)
    else:
        token = re.sub(r"(.)\1{2,}", r"\1", token)

    # keep only alphanumerics
    token = re.sub(r"[^a-z0-9]", "", token)

    return token.strip()


def split_punctuation(text):
    if not text:
        return []
    # Splits alphanumerics from specific punctuation marks
    return re.findall(r"[A-Za-z0-9]+|[.,?!]", text)


def tokenize(text):
    if not text or not text.strip():
        return []

    raw = split_punctuation(text)
    tokens = []

    i = 0
    while i < len(raw):
        tok = raw[i]

        # Check if the token is punctuation
        if tok in {".", ",", "?", "!"}:
            # Calculate the length of the punctuation sequence (run_len)
            run_len = 0
            while (i + run_len < len(raw)) and (raw[i + run_len] == tok):
                run_len += 1

            # Apply specific rules based on the punctuation type
            if tok == ",":
                # Rule: Always skip commas
                pass
            elif tok == "!":
                # Rule: Keep only if single instance, skip if repeated (!!!)
                if run_len == 1:
                    tokens.append(tok)
            elif tok == "?":
                # Rule: Keep all instances (e.g. ?? -> ?, ?)
                tokens.extend([tok] * run_len)
            elif tok == ".":
                # Rule: Keep all instances (e.g. ... -> ., ., .)
                tokens.extend([tok] * run_len)

            # Skip ahead by the run length
            i += run_len
        else:
            # It is a word/alphanumeric sequence
            cleaned = clean_token(tok)
            if cleaned:
                tokens.append(cleaned)
            i += 1

    return tokens


def pad_tokens(tokens, start_count=2):
    if not tokens:
        return ["<START>", "<START>", "<END>"]
    return ["<START>"] * start_count + tokens + ["<END>"]


def map_unknowns(tokens, vocab):
    return [t if t in vocab or t in {"<START>", "<END>", "<UNK>"} else "<UNK>" for t in tokens]


def choose_from_distribution(counts_dict):
    if not counts_dict:
        return None

    total = sum(counts_dict.values())
    if total == 0:
        return None

    r = random.random()
    cumulative = 0.0

    for word, count in counts_dict.items():
        cumulative += count / total
        if r <= cumulative:
            return word

    return random.choice(list(counts_dict.keys()))
