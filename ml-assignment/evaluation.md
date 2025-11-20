# Evaluation

Please provide a 1-page summary of your design choices for the Trigram Language Model.

This should include:

- How you chose to store the n-gram counts.
- How you handled text cleaning, padding, and unknown words.
- How you implemented the `generate` function and the probabilistic sampling.
- Any other design decisions you made and why you made them.

**Trigram Language Model — Design Choices**
-------------------------------------------

### **1\. N-Gram Storage Structure**

I chose to store trigram counts using a 3-level nested defaultdict:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   self.trigram_counts[w1][w2][w3] += 1   `

This structure allows constant-time updates while training and provides clean, intuitive access when generating text. It avoids key-existence checks and keeps the implementation compact. The hierarchical dictionary representation matches the mathematical structure of trigram models, where each pair (w1, w2) defines a conditional distribution over w3.

### **2\. Text Cleaning & Normalization**

Text is cleaned in a preprocessing pipeline implemented in utils.py.The cleaning logic was guided by both assignment objectives and the extended test suite.

Key decisions:

#### **a. Lowercasing**

All tokens are converted to lowercase to reduce vocabulary size and improve consistency.

#### **b. Repeated-Letter Handling**

I implemented a two-step rule:

*   Collapse **3+ repeated letters** down to **1** (e.g., "Heeellp" → "help").
    
*   Preserve double letters that are naturally part of English words ("Hello" → "hello", NOT "helo").
    

This balances test-driven behavior and realistic text normalization.

#### **c. Punctuation Handling**

Using split\_punctuation(), words and punctuation are separated. Then:

*   End-of-word punctuation such as "!!!" and "," is removed.
    
*   Single-character "?" and "!" are preserved if they stand alone (matches test expectations).
    

This yields clean tokens while preserving meaningful punctuation.

#### **d. Symbol & Noise Removal**

Non-alphanumeric characters are removed after segmentation to ensure robustness against noisy inputs like "@#$", "!!!", "???", and unicode variants.

### **3\. Tokenization**

The tokenizer performs:

1.  Whitespace trimming
    
2.  Regex-based word + punctuation extraction
    
3.  Cleaning each token
    
4.  Conditional punctuation removal
    
5.  Final clean token list returned
    

The tokenizer is built to pass all 26 test cases, including unicode, numbers, repeated letters, noise, and multi-punctuation sequences.

### **4\. Padding Strategy**

To allow trigram learning at sentence boundaries, I prepend two tokens and append one token:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML  `w1 w2 ... wn` 

This ensures:

*   deterministic generation start
    
*   correct formation of initial trigrams
    
*   clean sentence termination
    

### **5\. Unknown Words**

During generation, tokens are mapped to known words via map\_unknowns():

*   Known words → unchanged
    
*   Unknown words →
    
*   Special tokens are preserved
    

This ensures the model remains robust even when encountering unseen vocabulary.

### **6\. Generation & Sampling**

The generate() function:

1.  Starts from (, )
    
2.  Looks up the trigram distribution for (w1, w2)
    
3.  Uses **probabilistic sampling**, not greedy selection:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   choose_from_distribution(counts_dict)   `

The sampling helper converts raw counts into normalized cumulative probabilities and selects a word proportionally. This avoids deterministic outputs and produces varied sentences.

Generation stops when:

*   is sampled
    
*   No continuations exist
    
*   Max length is reached
    

This prevents infinite loops and matches assignment constraints.

### **7\. Additional Design Decisions**

*   All preprocessing and helpers are placed in utils.py to keep ngram\_model.py clean and modular.
    
*   The model is resilient to noisy inputs (symbols, unicode, long repetition, malformed punctuation).
    
*   The implementation adheres to the project structure expected by pytest and the assignment.
    

### **8\. Summary**

The final design balances simplicity, correctness, modularity, and robustness.passes all **26 functional tests**, including edge-case and stress tests, while remaining faithful to the classic trigram language-modeling approach.
