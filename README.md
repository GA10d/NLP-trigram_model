# Trigram Language Model for Natural Language Processing
This repository contains a trigram language model implementation for natural language processing tasks, including n-gram extraction, probability calculation, text generation, and essay scoring classification. The model is developed as part of the **COMS W4705 - Natural Language Processing (Fall 2025)** course homework.


## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Key Features](#key-features)
- [Installation & Dependencies](#installation--dependencies)
- [Usage Guide](#usage-guide)
- [Sample Run Results](#sample-run-results)
- [Core Components Explanation](#core-components-explanation)
- [Author](#author)


## Project Overview
The trigram language model leverages statistical methods to model text sequences. It supports:
1. Extraction of unigrams, bigrams, and trigrams from text.
2. Calculation of raw (unsmoothed) and smoothed (linear interpolation) probabilities for n-grams.
3. Random text generation based on learned trigram patterns.
4. Perplexity calculation to evaluate model performance on text corpora.
5. Essay scoring classification (high/low proficiency) using perplexity as a metric.


## Directory Structure
Before running the script, ensure your project follows this directory layout (critical for data loading):
```
project-name/
├── data/                  # Corpus and essay data directory
│   ├── brown_test.txt     # Test corpus (Brown corpus subset)
│   ├── brown_train.txt    # Training corpus (Brown corpus subset)
│   └── ets_toefl_data/    # TOEFL essay scoring data
│       ├── test_high/     # High-proficiency test essays (multiple .txt files)
│       ├── test_low/      # Low-proficiency test essays (multiple .txt files)
│       ├── train_high.txt # High-proficiency training essays
│       └── train_low.txt  # Low-proficiency training essays
├── trigram_model.py       # Core trigram model code
└── README.md              # Project documentation (this file)
```


## Key Features
| Feature | Description |
|---------|-------------|
| **n-gram Extraction** | Converts text sequences into unigrams, bigrams, or trigrams with `START`/`STOP` padding (e.g., trigrams for a 3-word sentence include 4 padded trigrams). |
| **Raw Probability Calculation** | Computes unsmoothed probabilities for unigrams, bigrams, and trigrams (handles zero counts with fallback to lower-order n-gram probabilities). |
| **Smoothed Probability** | Uses linear interpolation (λ₁=λ₂=λ₃=1/3) to avoid zero probabilities and improve generalization. |
| **Text Generation** | Generates random sentences up to 20 tokens (stops early if `STOP` is generated). |
| **Perplexity Evaluation** | Measures model performance (lower perplexity = better model fit to the corpus). |
| **Essay Classification** | Distinguishes high/low proficiency essays by comparing perplexity against two specialized models (high/low training data). |


## Installation & Dependencies
The code uses only Python's standard libraries—**no external dependencies** are required. Ensure you have:
- Python 3.6 or higher (tested on Python 3.8+).


## Usage Guide
### 1. Prepare Data
Follow the [directory structure](#directory-structure) to place your corpus and essay data in the `data/` folder. The Brown corpus (train/test) and TOEFL essay data are required for full functionality.

### 2. Run the Script
Execute the script interactively (recommended for testing) or non-interactively:
```bash
# Run interactively (keeps Python prompt open for further testing)
python -i trigram_model.py

# Run non-interactively (prints all results and exits)
python trigram_model.py
```

### 3. Key Functions to Test (Interactive Mode)
After running `python -i trigram_model.py`, you can test core functions manually:
```python
# Example 1: Extract trigrams from a custom sentence
get_ngrams(["I", "love", "NLP"], 3)

# Example 2: Create a model with custom training data
custom_model = TrigramModel("data/custom_train.txt")

# Example 3: Calculate perplexity on a custom test file
custom_model.perplexity("data/custom_test.txt")

# Example 4: Generate a new sentence
custom_model.generate_sentence(t=15)  # Max 15 tokens
```


## Sample Run Results
Below is the output from a full run of `trigram_model.py` (using the Brown corpus and TOEFL essay data):

### Part 1: n-gram Extraction
Extracts padded n-grams for the sentence `["Guo", "Zhe", "Wen"]`:
```
Part 1 - Extracting n-grams from a Sentence
[('Guo',), ('Zhe',), ('Wen',), ('STOP',)]  # Unigrams
[('START', 'Guo'), ('Guo', 'Zhe'), ('Zhe', 'Wen'), ('Wen', 'STOP')]  # Bigrams
[('START', 'START', 'Guo'), ('START', 'Guo', 'Zhe'), ('Guo', 'Zhe', 'Wen'), ('Zhe', 'Wen', 'STOP')]  # Trigrams
```

### Part 2: n-gram Counts (Brown Training Corpus)
Counts of common n-grams in `brown_train.txt`:
```
Part 2 - Counting n-grams in a Corpus
trigramcounts:  5478  # Count of ('START', 'START', 'the')
bigramcounts:  5478   # Count of ('START', 'the')
unigramcounts:  61428 # Count of ('the',)
```

### Interlude: Text Generation
3 random sentences generated by the Brown corpus-trained model:
```
Interlude - Generating text
senctence1:  ['but', 'he', 'had', 'long', 'black', 'hair', '.', 'STOP']
senctence2:  ['one', 'of', 'them', 'covering', 'various', 'contingencies', '-', 'are', 'not', 'reflected', 'in', 'a', 'manner', 'typically', 'continental', '.', 'STOP']
senctence3:  ['dictionary', 'forms', 'found', 'in', 'the', 'boundary', 'of', 'civil', 'defense', 'setup', 'and', 'begin', 'to', 'assert', 'himself', ',', 'surprised', ',', 'UNK', 'on']
```

### Part 6: Perplexity Evaluation
Perplexity of the model on training vs. test corpora (lower = better fit):
```
Part 6 - Perplexity
Perplexity of the training corpus:  16.69002032056936
Perplexity of the testing corpus:  220.07056728158165
```
*Note: Higher perplexity on test data is expected (generalization gap).*

### Part 7: Essay Scoring Classification
Accuracy of distinguishing high/low proficiency TOEFL essays:
```
Part 7 - Using the Model for Text Classification
Training on corpus: data/ets_toefl_data/train_high.txt
Training on corpus: data/ets_toefl_data/train_low.txt
Accuracy of essay scoring experiment:  0.8386454183266933
```
*~84% accuracy indicates strong performance for this task.*


## Core Components Explanation
### 1. `corpus_reader(corpusfile, lexicon)`
- Reads a text file line-by-line, lowercases text, and splits into tokens.
- If a `lexicon` is provided, replaces out-of-lexicon tokens with `UNK` (unknown).

### 2. `get_ngrams(sequence, n)`
- Pads sequences with `n-1` `START` tokens (for context) and 1 `STOP` token (end of sentence).
- Returns a list of n-gram tuples (e.g., trigrams for `n=3`).

### 3. `TrigramModel` Class
- **`__init__`**: Builds a lexicon (tokens with count > 1) and initializes n-gram counting.
- **`count_ngrams`**: Populates `unigramcounts`, `bigramcounts`, and `trigramcounts` from the corpus.
- **`raw_*_probability`**: Computes unsmoothed probabilities for unigrams, bigrams, trigrams.
- **`smoothed_trigram_probability`**: Applies linear interpolation to trigram probabilities.
- **`sentence_logprob`**: Calculates the log probability of a sentence (sums log probabilities of its trigrams).
- **`perplexity`**: Computes perplexity (2^(-average log probability)) for a corpus.
- **`generate_sentence`**: Generates random sentences using trigram frequency weights.

### 4. `essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2)`
- Trains two models: one on high-proficiency essays (`training_file1`), one on low-proficiency (`training_file2`).
- Tests each model on high/low test essays: classifies an essay as "high" if the high-model perplexity is lower (and vice versa).
- Returns classification accuracy.


## Author
- Zhewen Guo (zg2567)
- Course: COMS W4705 - Natural Language Processing (Fall 2025)
