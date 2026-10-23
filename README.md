
# Enterprise Knowledge Assistant

A local AI-powered knowledge assistant that uses
Retrieval-Augmented Generation (RAG) to answer questions
from enterprise documents.

## Project Goals

- Load and process PDF, TXT, and Markdown documents
- Generate text embeddings for semantic search
- Store document vectors using ChromaDB
- Generate grounded answers using a local LLM
- Display supporting document passages

## Technology Stack

- Python
- Streamlit
- Ollama
- Sentence Transformers
- ChromaDB
- PyPDF

## Architecture

Documents -> Text Extraction -> Chunking -> Embeddings
-> Vector Database -> Retrieval -> Local LLM -> Answer

## Prerequisites

- Python 3.11 or 3.12
- Git
- Ollama

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies.
4. Download the selected Ollama model.
5. Add permitted sample documents.
6. Start the application.

## Run Locally

```powershell
python -m streamlit run app.py
```

## Privacy

Use synthetic or authorized documents only.
Local processing does not replace appropriate access
controls or security practices.

## Project Status

In development.
