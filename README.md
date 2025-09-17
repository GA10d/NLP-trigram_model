# Trigram Language Model for Natural Language Processing (English Version)
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

# 用于自然语言处理的三元语言模型（中文版）
本仓库包含一个三元语言模型实现，适用于自然语言处理任务，包括n-gram提取、概率计算、文本生成和作文评分分类。该模型是作为**COMS W4705 - 自然语言处理（2025年秋季学期）** 课程作业的一部分开发的。


## 目录
- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [主要功能](#主要功能)
- [安装与依赖](#安装与依赖)
- [使用指南](#使用指南)
- [示例运行结果](#示例运行结果)
- [核心组件说明](#核心组件说明)
- [作者](#作者)


## 项目概述
三元语言模型利用统计方法对文本序列进行建模。它支持：
1. 从文本中提取一元组（unigrams）、二元组（bigrams）和三元组（trigrams）。
2. 计算n-gram的原始（未平滑）和平滑（线性插值）概率。
3. 基于学习到的三元模式生成随机文本。
4. 计算困惑度（perplexity）以评估模型在文本语料上的性能。
5. 使用困惑度作为指标进行作文评分分类（高/低水平）。


## 目录结构
运行脚本前，请确保项目遵循以下目录布局（这对数据加载至关重要）：
```
project-name/
├── data/                  # 语料库和作文数据目录
│   ├── brown_test.txt     # 测试语料库（Brown语料库子集）
│   ├── brown_train.txt    # 训练语料库（Brown语料库子集）
│   └── ets_toefl_data/    # TOEFL作文评分数据
│       ├── test_high/     # 高水平测试作文（多个.txt文件）
│       ├── test_low/      # 低水平测试作文（多个.txt文件）
│       ├── train_high.txt # 高水平训练作文
│       └── train_low.txt  # 低水平训练作文
├── trigram_model.py       # 核心三元模型代码
└── README.md              # 项目文档（本文档）
```


## 主要功能
| 功能 | 描述 |
|------|------|
| **n-gram提取** | 将文本序列转换为一元组、二元组或三元组，并添加`START`/`STOP`填充（例如，一个3词句子的三元组包含4个带填充的三元组）。 |
| **原始概率计算** | 计算一元组、二元组和三元组的未平滑概率（通过退回到低阶n-gram概率处理零计数问题）。 |
| **平滑概率** | 使用线性插值（λ₁=λ₂=λ₃=1/3）避免零概率，提高泛化能力。 |
| **文本生成** | 生成最多20个 token 的随机句子（如果生成`STOP`则提前停止）。 |
| **困惑度评估** | 衡量模型性能（困惑度越低，模型对语料的拟合越好）。 |
| **作文分类** | 通过比较两个专用模型（高/低水平训练数据）的困惑度来区分高/低水平作文。 |


## 安装与依赖
代码仅使用Python的标准库——**无需外部依赖**。请确保您已安装：
- Python 3.6或更高版本（已在Python 3.8+上测试）。


## 使用指南
### 1. 准备数据
按照[目录结构](#目录结构)将您的语料库和作文数据放在`data/`文件夹中。Brown语料库（训练/测试）和TOEFL作文数据是实现完整功能所必需的。

### 2. 运行脚本
以交互方式（推荐用于测试）或非交互方式执行脚本：
```bash
# 交互方式运行（保持Python提示符打开以便进一步测试）
python -i trigram_model.py

# 非交互方式运行（打印所有结果后退出）
python trigram_model.py
```

### 3. 可测试的主要函数（交互模式）
运行`python -i trigram_model.py`后，您可以手动测试核心函数：
```python
# 示例1：从自定义句子中提取三元组
get_ngrams(["I", "love", "NLP"], 3)

# 示例2：使用自定义训练数据创建模型
custom_model = TrigramModel("data/custom_train.txt")

# 示例3：计算自定义测试文件的困惑度
custom_model.perplexity("data/custom_test.txt")

# 示例4：生成新句子
custom_model.generate_sentence(t=15)  # 最多15个token
```


## 示例运行结果
以下是`trigram_model.py`完整运行的输出（使用Brown语料库和TOEFL作文数据）：

### 第1部分：n-gram提取
提取句子`["Guo", "Zhe", "Wen"]`的带填充n-gram：
```
第1部分 - 从句子中提取n-gram
[('Guo',), ('Zhe',), ('Wen',), ('STOP',)]  # 一元组
[('START', 'Guo'), ('Guo', 'Zhe'), ('Zhe', 'Wen'), ('Wen', 'STOP')]  # 二元组
[('START', 'START', 'Guo'), ('START', 'Guo', 'Zhe'), ('Guo', 'Zhe', 'Wen'), ('Zhe', 'Wen', 'STOP')]  # 三元组
```

### 第2部分：n-gram计数（Brown训练语料库）
`brown_train.txt`中常见n-gram的计数：
```
第2部分 - 语料库中的n-gram计数
trigramcounts:  5478  # ('START', 'START', 'the')的计数
bigramcounts:  5478   # ('START', 'the')的计数
unigramcounts:  61428 # ('the',)的计数
```

### 插曲：文本生成
由Brown语料库训练的模型生成的3个随机句子：
```
插曲 - 文本生成
句子1:  ['but', 'he', 'had', 'long', 'black', 'hair', '.', 'STOP']
句子2:  ['one', 'of', 'them', 'covering', 'various', 'contingencies', '-', 'are', 'not', 'reflected', 'in', 'a', 'manner', 'typically', 'continental', '.', 'STOP']
句子3:  ['dictionary', 'forms', 'found', 'in', 'the', 'boundary', 'of', 'civil', 'defense', 'setup', 'and', 'begin', 'to', 'assert', 'himself', ',', 'surprised', ',', 'UNK', 'on']
```

### 第6部分：困惑度评估
模型在训练语料与测试语料上的困惑度（越低表示拟合越好）：
```
第6部分 - 困惑度
训练语料的困惑度:  16.69002032056936
测试语料的困惑度:  220.07056728158165
```
*注：测试数据的困惑度较高是正常的（存在泛化差距）。*

### 第7部分：作文评分分类
区分高/低水平TOEFL作文的准确率：
```
第7部分 - 使用模型进行文本分类
在语料上训练：data/ets_toefl_data/train_high.txt
在语料上训练：data/ets_toefl_data/train_low.txt
作文评分实验的准确率:  0.8386454183266933
```
*~84%的准确率表明该任务的性能良好。*


## 核心组件说明
### 1. `corpus_reader(corpusfile, lexicon)`
- 逐行读取文本文件，将文本转为小写并分割为token。
- 如果提供了`lexicon`，则将词汇表外的token替换为`UNK`（未知）。

### 2. `get_ngrams(sequence, n)`
- 用`n-1`个`START` token（提供上下文）和1个`STOP` token（句子结束）填充序列。
- 返回n-gram元组列表（例如，`n=3`时返回三元组）。

### 3. `TrigramModel`类
- **`__init__`**：构建词汇表（计数>1的token）并初始化n-gram计数。
- **`count_ngrams`**：从语料库中填充`unigramcounts`、`bigramcounts`和`trigramcounts`。
- **`raw_*_probability`**：计算一元组、二元组、三元组的未平滑概率。
- **`smoothed_trigram_probability`**：对三元组概率应用线性插值。
- **`sentence_logprob`**：计算句子的对数概率（求和其三元组的对数概率）。
- **`perplexity`**：计算语料库的困惑度（2^(-平均对数概率））。
- **`generate_sentence`**：使用三元组频率权重生成随机句子。

### 4. `essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2)`
- 训练两个模型：一个基于高水平作文（`training_file1`），一个基于低水平作文（`training_file2`）。
- 在高/低水平测试作文上测试每个模型：如果高水平模型的困惑度更低，则将作文分类为“高水平”（反之亦然）。
- 返回分类准确率。


## 作者
- 郭哲文（zg2567）
- 课程：COMS W4705 - 自然语言处理（2025年秋季学期）
