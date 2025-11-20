# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model.

## How to Run

You can run, train, and generate text using the trigram model with the following commands.

### 1. Manual Model Usage (Python)

```python
from src.ngram_model import TrigramModel

model = TrigramModel()
model.fit("This is a sample text. This is another example sentence.")
print(model.generate())
```

### 2. Running All Tests

```bash
pytest
```

### 3. Running Only the Required Tests

```bash
pytest tests/test_ngram.py
```

### 4. Running Extended Tests

```bash
pytest tests/test_full_trigram.py
```

### 5. Project Structure

```text
ml-assignment/
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   └── ngram_model.py
│
├── tests/
│   ├── test_ngram.py
│   └── test_full_trigram.py
│
├── README.md
└── evaluation.md
```

### 6. Installation Instructions

```bash
pip install -r requirements.txt
```

### 7. Example CLI Invocation (if you add a generate.py script)

```bash
python generate.py --input "hello world"
```

---


## Design Choices

Please document your design choices in the `evaluation.md` file. This should be a 1-page summary of the decisions you made and why you made them.
