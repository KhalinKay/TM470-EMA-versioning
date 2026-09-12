# RAG Study Assistant

A local, fully offline Retrieval-Augmented Generation (RAG) chatbot for studying from your own documents (PDF, `.txt`, `.docx`). No document content or questions are ever sent to an external API. All embedding and generation runs through a locally hosted [Ollama](https://ollama.com/) server.

> **EMA report:** the full written submission is [report/EMA_L2049806_TM470.docx](report/EMA_L2049806_TM470.docx). Everything else in this repository is the supporting application code, tests and evaluation harness described in that report.

## Features

- Upload PDF, Word (`.docx`) or plain text study documents through a Gradio web UI.
- Documents are chunked (512 characters / 64 overlap by default) and embedded locally with `nomic-embed-text`, then indexed with FAISS for fast similarity search.
- Answers are generated with a local Llama 3 8B model via Ollama, grounded strictly in the retrieved excerpts.
- Every answer includes a `Source: [filename], page [n]` citation.
- Short-term conversation memory (last 4 turns / ~1500 tokens) allows natural follow-up questions.
- A TruLens-based evaluation harness scores the RAG Triad (Groundedness, Answer Relevance, Context Relevance) across multiple chunk sizes, using a local Ollama model as the judge.

## Architecture

```mermaid
flowchart LR
    U[User] -->|uploads docs| UI[Gradio UI]
    UI --> ING[Ingestion: loaders + splitter]
    ING --> EMB[OllamaEmbeddings]
    EMB --> IDX[(FAISS index)]
    U -->|asks question| UI
    UI --> RET[Retriever: top-k similarity search]
    IDX --> RET
    RET --> GEN[ChatOllama generation]
    GEN --> UI
    UI --> U
```

## Prerequisites

1. Download and install Ollama from [ollama.com/download](https://ollama.com/download) (Windows, macOS and Linux are all supported). The installer starts the Ollama server automatically in the background on `http://localhost:11434`; no separate setup step is needed.
2. Open a terminal and confirm the install worked:
   ```powershell
   ollama --version
   ```
3. Pull the two models this application needs. This is a one-off download of roughly 5 GB in total, so it can take a few minutes depending on connection speed:
   ```powershell
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```
4. Install [Python](https://www.python.org/downloads/) 3.11 or later, if not already installed.

No GPU is required. Both models run on CPU; generation is simply faster with one.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Adjust `.env` if your Ollama server, models, or chunking parameters differ from the defaults.

## Running the app

```powershell
python main.py
```

This launches the Gradio UI (default `http://127.0.0.1:7860`). Upload one or more documents, wait for the "Indexed..." status message, then ask questions in the chat box.

If the app reports it cannot reach Ollama, open a terminal and run `ollama serve` to start it manually, then reload the page. If a model name is reported as not found, re-run the `ollama pull` commands from the Prerequisites section above.

## Running tests

Unit tests use lightweight fake embeddings/LLM implementations, so they run without a live Ollama server:

```powershell
pytest
```

## Running the chunk-size evaluation

1. Place representative sample documents in `data/sample_docs/`.
2. Run:
   ```powershell
   python -m evaluation.run_evaluation --sample-docs data/sample_docs
   ```
3. Results are written to `data/eval_results/chunk_size_comparison.csv` and `.png`, comparing Groundedness, Answer Relevance and Context Relevance across chunk sizes 256, 512 and 1024 characters. This requires Ollama to be running (the same local model is used as the judge via TruLens's LiteLLM provider).

## Project structure

```
report/              EMA_L2049806_TM470.docx (the submission) and the script that builds it
src/rag_assistant/   application package (ingestion, indexing, memory, prompts, pipeline, ui)
evaluation/          TruLens RAG Triad evaluation harness and test question set
tests/                unit tests (offline, using fake embeddings/LLM)
data/                 sample documents, FAISS index storage, evaluation results (gitignored except .gitkeep)
```
