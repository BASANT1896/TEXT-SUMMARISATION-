# Hybrid Extractive-Abstractive Text Summarization Using TF-IDF and Fine-Tuned FLAN-T5

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces) 
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper: **"Hybrid Extractive-Abstractive Text Summarization Using TF-IDF and Fine-Tuned FLAN-T5"** authored by *Utkarsh Kumar, Basant, and Dr. Gaurav Singal* at the **Netaji Subhas University of Technology (NSUT)**, New Delhi.

The project features a full-stack, low-latency text summarization pipeline deployed as an asynchronous web application via **FastAPI** and hosted on the **Hugging Face Hub**.

---

#### Quantitative Evaluation (ROUGE Metrics)
[cite_start]Evaluated on a completely isolated 1,000-sample test set, the hybrid framework yields competitive, factually grounded scores across the industry standard evaluation suite[cite: 113, 123]:

| Metric | Testing Score (%) | Targeted Core Assessment Area |
| :--- | :--- | :--- |
| **ROUGE-1** | **28.30%** | [cite_start]Overlap of unigrams (capturing individual key facts) [cite: 122, 132] |
| **ROUGE-2** | **11.96%** | [cite_start]Overlap of bigrams (word order and structural fluency) [cite: 122, 136] |
| **ROUGE-L** | **21.74%** | [cite_start]Longest Common Subsequence (sentence-level structure) [cite: 122, 141] |

* [cite_start]**ROUGE-1:** Measures the overlap of single words (unigrams) between the generated and reference summaries to evaluate how well individual key facts are captured[cite: 121, 122].
* [cite_start]**ROUGE-2:** Measures the overlap of two-word phrases (bigrams) to assess structural fluency and localized word ordering[cite: 121, 122].
* [cite_start]**ROUGE-L:** Measures the Longest Common Subsequence (LCS) to evaluate sentence-level structure and structural coherence without requiring strictly consecutive word matches[cite: 122].

## 🚀 Key Features
* **"Extract-then-Abstract" Pipeline:** Combines traditional statistical reliability with modern generative transformer fluency to eliminate large language model (LLM) hallucinations.
* **Context Window Optimization:** Uses a dynamic TF-IDF preliminary gatekeeper to strip out background noise and extract the top $K$ salient sentences, squeezing dense multi-page text into an optimized input tensor.
* **Multi-Format Parsing:** Integrated file parsers supporting real-time processing of user-uploaded **PDF, DOCX, and TXT** documents.
* **Production Deployment:** Deployed with high performance using a Python-based asynchronous FastAPI backend.

---


### 💻 Tech Stack & Infrastructure
* **Deep Learning Backend:** PyTorch [cite: 107], HuggingFace (Transformers & Datasets) [cite: 107]
* **Statistical NLP Gatekeeping:** Scikit-Learn [cite: 108], NLTK (Natural Language Toolkit) [cite: 108]
* **API Delivery Layer:** FastAPI [cite: 109], Uvicorn ASGI Web Server [cite: 109]
* **Document Handling Integration:** PyPDF2 [cite: 102], python-docx [cite: 102], Regular Expressions (Regex) [cite: 90]
