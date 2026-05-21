
# Intelligent Document Assistant

Hybrid Extractive and LLM-Based PDF Summarisation System

---

## Overview

The Intelligent Document Assistant is an offline AI system for summarising PDF documents using a hybrid approach that combines classical Natural Language Processing (NLP) techniques with Large Language Models (LLMs). It supports both extractive summarisation (TF-IDF with Maximum Marginal Relevance) and abstractive summarisation using a locally deployed LLM via Ollama. A graphical user interface (GUI) is provided for single-document and batch processing with real-time execution feedback.

---

## Key Features

* Fully offline operation (no cloud APIs)
* Extractive summarisation using TF-IDF + MMR
* Abstractive summarisation using a local LLM (Ollama)
* Graphical User Interface (GUI) for usability
* Single-document and batch processing
* Runtime performance measurement and comparison
* Lightweight hallucination detection
* Structured output files for analysis

---

## System Architecture

The system consists of four main components:

1. GUI Layer – User interaction and execution control (Tkinter)
2. Preprocessing Layer – PDF text extraction and cleaning
3. Summarisation Layer – Extractive (TF-IDF + MMR) and abstractive (LLM) methods
4. Evaluation and Output Layer – Runtime analysis, hallucination checks, and result generation

---

## Project Structure

```
doc_ai_assistant/
├── src/
│   ├── main.py
│   ├── batchrun.py
│   ├── preprocess.py
│   ├── extractive.py
│   └── llm.py
├── gui.py
├── data/
│   ├── input/
│   └── output/
├── experiment_results.csv
└── README.md
```

---

## Requirements

* Python 3.10+
* Ollama (installed locally)
* Python libraries: scikit-learn, numpy, PyPDF2 or pdfplumber, tkinter

---

## How to Run

GUI (recommended):

```
python gui.py
```

Single document:

```
python src/main.py
```

Batch processing:

```
python src/batchrun.py
```

Place input PDFs in:

```
data/input/
```

---

## Outputs

For each document, the system generates:

* Cleaned text
* Extractive summary
* LLM-based abstractive summary
* Comparison file
* Runtime and evaluation metrics

Outputs are saved in:

```
data/output/
```

---

## Notes

* Extractive summarisation is extremely fast but less readable.
* LLM-based summarisation produces more coherent summaries at higher computational cost.
* The system is intended as a decision-support tool, not an authoritative summariser.

---

Developed for the Artificial Intelligence Foundations module

---
