# RAG — Retrieval-Augmented Generation

A practical **Retrieval-Augmented Generation (RAG)** implementation built to understand and practice the core RAG pipeline from document processing to LLM-based answer generation.

## Overview

This project implements a basic RAG pipeline using:

* Document chunking
* Text embeddings
* FAISS vector search
* Semantic retrieval
* Context construction
* RAG prompting
* Local LLM generation with Ollama

### Pipeline

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Index
   ↓
Query Embedding
   ↓
Retriever
   ↓
Relevant Context
   ↓
RAG Prompt
   ↓
LLM
   ↓
Generated Answer
```

## Project Structure

```text
RAG/
├── data/
│   └── document.txt
├── build_index.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── context.py
├── prompt.py
├── generator.py
├── query.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama and pull the LLM used by the project.

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

## Run the Project

### Build the vector index

```bash
python build_index.py
```

This performs:

```text
Document
   ↓
Chunking
   ↓
Embedding
   ↓
FAISS Index
```

### Run the RAG query pipeline

```bash
python query.py
```

Enter a question related to the document.

Example:

```text
What is the purpose of an Industrial Energy Management System?
```

The system retrieves relevant document chunks and passes them as context to the LLM to generate an answer.

## Current Learning Scope

This project is being developed as a practical study of RAG concepts, including:

* Chunking and chunk overlap
* Text embeddings
* Cosine similarity
* FAISS vector indexing
* Semantic search
* Top-K retrieval
* Context construction
* RAG prompt design
* LLM generation
* Retrieval evaluation

## Status

**Core RAG pipeline implemented and tested.**

Next focus:

```text
Retrieval Evaluation
        ↓
Top-K Comparison
        ↓
Similarity Thresholds
        ↓
RAG Optimization
```
