# Hybrid Extractive-Abstractive Text Summarization Using TF-IDF and Fine-Tuned FLAN-T5

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces) 
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: **"Hybrid Extractive-Abstractive Text Summarization Using TF-IDF and Fine-Tuned FLAN-T5"** authored by *Utkarsh Kumar, Basant, and Dr. Gaurav Singal* at the **Netaji Subhas University of Technology (NSUT)**, New Delhi.

The project features a full-stack, low-latency text summarization pipeline deployed as an asynchronous web application via **FastAPI** and hosted on the **Hugging Face Hub**.

---

## 🚀 Key Features
* **"Extract-then-Abstract" Pipeline:** Combines traditional statistical reliability with modern generative transformer fluency to eliminate large language model (LLM) hallucinations.
* **Context Window Optimization:** Uses a dynamic TF-IDF preliminary gatekeeper to strip out background noise and extract the top $K$ salient sentences, squeezing dense multi-page text into an optimized input tensor.
* **Multi-Format Parsing:** Integrated file parsers supporting real-time processing of user-uploaded **PDF, DOCX, and TXT** documents.
* **Production Deployment:** Deployed with high performance using a Python-based asynchronous FastAPI backend.

---

## 🧬 System Architecture & Methodology

The application relies on a synergistic two-stage hybrid framework designed to bound the creative generation of an abstractive model with strict, statistically verified factual context:

### 🧬 Methodology & Evaluation Metrics

#### 1. Extractive Phase (TF-IDF)
[cite_start]The cumulative relevance weight $W_{i,j}$ for any given term $i$ in a sentence $j$ is calculated dynamically to rank and filter the top $K$ (default optimum threshold $K=5$) most critical facts[cite: 65, 72, 73]:

$$W_{i,j} = tf_{i,j} \times \log\left(\frac{N}{df_{i}}\right)$$

[cite_start]Where $tf_{i,j}$ represents the raw frequency of term $i$ in sentence $j$, $N$ is the total number of sentences in the document, and $df_{i}$ is the number of sentences containing term $i$[cite: 77]. [cite_start]A cumulative salience score is computed for each sentence by summing the weights of its constituent terms, allowing the system to rank and extract the top $K$ sentences to form the context vector $X$[cite: 78].

#### 2. Abstractive Phase (Sequence Generation)
[cite_start]The heavily filtered context block vector $X$ is fed into a fine-tuned `FLAN-T5-base` model[cite: 79]. [cite_start]The network computes sequence distribution metrics by maximizing conditional probability over target sequence slices[cite: 80]: 

$$P(Y|X)=\prod_{t=1}^{m}P(y_{t}|y_{<t},X)$$

* **Decoding Strategy:** Instead of greedy decoding, the model utilizes Beam Search decoding to explore multiple sequence hypotheses simultaneously (beam width of 4)[cite: 83, 95]. To prevent the generation of disjointed or excessively repetitive text, the generation is mathematically constrained using dynamic length penalties and a sequence repetition penalty of 1.3[cite: 84, 95].

---

### 📊 Fine-Tuning Setup & Performance

The underlying `FLAN-T5-base` model was fine-tuned on **8,000 samples** of the standard **CNN/DailyMail dataset** for 3 complete training epochs using PyTorch and Hugging Face acceleration[cite: 30, 112, 114].

#### Training Configuration
* **Optimizer Learning Rate:** `3e-5` paired with a weight decay of `0.01` to prevent overfitting[cite: 115].
* **Batch Sizing Strategy:** Per-device training batch size of 2, coupled with a gradient accumulation step of 4 to effectively simulate a larger batch size stable gradient descent[cite: 114].

#### Quantitative Evaluation (ROUGE Metrics)
Evaluated on a completely isolated 1,000-sample test set, the hybrid framework yields competitive, factually grounded scores across the industry standard evaluation suite[cite: 113, 123]:

| Metric | Testing Score (%) | Targeted Core Assessment Area |
| :--- | :--- | :--- |
| **ROUGE-1** | **28.30%** | Overlap of unigrams (capturing individual key facts) [cite: 122, 132] |
| **ROUGE-2** | **11.96%** | Overlap of bigrams (word order and structural fluency) [cite: 122, 136] |
| **ROUGE-L** | **21.74%** | Longest Common Subsequence (sentence-level structure) [cite: 122, 141] |

---

### 💻 Tech Stack & Infrastructure
* **Deep Learning Backend:** PyTorch [cite: 107], HuggingFace (Transformers & Datasets) [cite: 107]
* **Statistical NLP Gatekeeping:** Scikit-Learn [cite: 108], NLTK (Natural Language Toolkit) [cite: 108]
* **API Delivery Layer:** FastAPI [cite: 109], Uvicorn ASGI Web Server [cite: 109]
* **Document Handling Integration:** PyPDF2 [cite: 102], python-docx [cite: 102], Regular Expressions (Regex) [cite: 90]
