# Trigram Language Model & Scaled Dot-Product Attention (Optional)

This repository contains the complete solution for the AI/ML Intern Assignment.  
It includes:

- A fully implemented **Trigram Language Model** (Task 1)
- An optional **Scaled Dot-Product Attention** module (Task 2)
- All preprocessing utilities
- A clean project structure
- Comprehensive tests (required + extended 26-case suite)

---

# Project Structure

```text
ml-assignment/
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   └── ngram_model.py
│
├── task2_attention/
│   ├── attention.py
│   └── demo_attention.py
│
├── tests/
│   ├── test_ngram.py
│   └── test_full_trigram.py
│
├── README.md
└── evaluation.md
```

# Installation

``` bash
pip install -r requirements.txt
```

## Requirements:

```
pytest
numpy   # only for Task 2
```

# Task 2 — Scaled Dot-Product Attention (Optional)

This module implements the attention mechanism used in Transformers, using only NumPy, as required.

## Running the Attention Demo (Task 2)

### Execute the demo:

``` bash
python task2_attention/demo_attention.py
```

You will see:

1. Q, K, V shapes

2. attention weight matrices

3. proper row normalization

4. masking behavior

## Task 2 Implementation Overview

``` cpp
Attention(Q, K, V) = softmax( (QKᵀ) / sqrt(d_k) ) V
```

# Key features:

1. Stable softmax

2. Scaled dot-product (division by √dₖ)

3. Supports batch dimensions

4. Supports masks (0 = masked out)

5. Produces correct attention weights + outputs
