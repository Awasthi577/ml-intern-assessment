# Evaluation

## Trigram Language Model — Design Choices

---

## 1. N-Gram Storage Structure

I chose to store trigram counts using a **3-level nested `defaultdict`**:

``` python
self.trigram_counts[w1][w2][w3] += 1

```
This structure:

- Allows constant-time updates while training  
- Avoids manual key-existence checks  
- Mirrors the mathematical definition of trigrams  
- Keeps the implementation simple and efficient  

Using a nested dictionary `(w1 → w2 → w3 → count)` makes accessing conditional distributions extremely clean when generating text.

---

## 2. Text Cleaning & Normalization

All preprocessing is implemented in `utils.py`.  
The cleaning pipeline is built to satisfy assignment requirements **and** all 26 tests.

### a. Lowercasing
All tokens are converted to lowercase to unify vocabulary and minimize token variations.

### b. Repeated-Letter Handling
A two-step normalization rule:

- Collapse **3 or more repeated letters** into **one**  
  - `"Heeellp"` → `"help"`
- Preserve valid double letters  
  - `"Hello"` → `"hello"` (NOT `"helo"`)

This ensures realistic English normalization while satisfying test expectations.

### c. Punctuation Handling

Using `split_punctuation()`, tokens are separated into words and punctuation.

Rules:

- Remove end-of-word punctuation sequences such as `"!!!"` or `","`
- Preserve **single-character** `"?"` and `"!"` when they appear alone  
- Remove multi-symbol noise like `"???"`, `"..."`, etc.

This produces clean, meaningful tokens without breaking sentence meaning.

### d. Symbol & Noise Removal

All non-alphanumeric symbols are stripped to ensure robustness against:

- Emoji  
- Random symbols (`@#$` etc.)  
- Excessive punctuation  
- Unicode oddities  

This keeps the vocabulary clean and consistent.

---

## 3. Tokenization

The tokenizer performs:

1. Whitespace trimming  
2. Regex-based word + punctuation extraction  
3. Token-level cleaning  
4. Conditional punctuation removal  
5. Assembly of the final cleaned token list  

It successfully handles:

- Unicode  
- Numbers  
- Repeated characters  
- Noisy sequences  
- Multi-punctuation patterns  

The tokenizer is fully validated by all 26 tests.

---

## 4. Padding Strategy

To teach the model sentence boundaries, I add:

```php
<START> <START> w1 w2 ... wn <END>
```

Padding ensures:

- Deterministic sentence start  
- Correct formation of initial trigrams  
- Predictable sentence endings  

This allows the generator to start and stop cleanly.

---

## 5. Unknown Words

Unknown words are handled by `map_unknowns()`:

- Known words → unchanged  
- Unknown words → `<UNK>`  
- Special tokens (`<START>`, `<END>`, `<UNK>`) remain stable  

This prevents model crashes on unseen tokens and maintains consistent generation behavior.

---

## 6. Generation & Sampling

The `generate()` function:

- Starts from:

```
(<START>, <START>)
```

- Retrieves the trigram distribution for `(w1, w2)`
- Samples the next word using **probabilistic sampling**, not greedy selection:

``` python
choose_from_distribution(counts_dict)
```

The sampling helper:

- Converts raw counts → probability distribution  
- Performs cumulative weighted sampling  
- Produces natural, non-deterministic output  

Generation stops if:

- `<END>` is sampled  
- No valid continuation exists  
- The maximum sentence length is exceeded  

This prevents infinite loops and creates natural-length sequences.

---

## 7. Additional Design Decisions

- All helpers are placed in `utils.py` to keep `ngram_model.py` clean and modular  
- The system is robust to:
  - Unicode text  
  - Noise symbols  
  - Long repeated patterns  
  - Edge-case punctuation  
- The project structure is fully compatible with pytest and the assignment  

---

## 8. Summary

The final design emphasizes:

- **Simplicity**
- **Correctness**
- **Modularity**
- **Robustness**

It passes **all 26 tests**, including stress tests and noise-handling scenarios, while remaining faithful to the classic trigram language-modeling paradigm.
