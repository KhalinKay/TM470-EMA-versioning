# RAG Study Assistant — Copilot Instructions

Local, fully offline Retrieval-Augmented Generation (RAG) study assistant chatbot.

## Stack
- Python 3.11
- LangChain (orchestration: loaders, splitter, retrieval chain)
- FAISS (vector index, CPU)
- Ollama (local inference server: Llama 3 8B generation, nomic-embed-text embeddings)
- Gradio (web UI)
- TruLens (RAG Triad evaluation: Answer Relevance, Context Relevance, Groundedness)

## Structure
- `src/rag_assistant/` — application package (ingestion, indexing, retrieval, memory, prompts, ui)
- `evaluation/` — TruLens evaluation harness and chunk-size comparison scripts
- `tests/` — unit tests
- `data/` — user-uploaded/sample documents and FAISS index storage (gitignored except `.gitkeep`)

## Conventions
- All processing must stay local; never call external/cloud APIs.
- Dependency versions are pinned in `requirements.txt`.
- Chunking default: 512 chars / 64 overlap (RecursiveCharacterTextSplitter), confirmed best by evaluation.
- Retrieval: top-4 chunks via FAISS cosine similarity.
- Every generated answer must include a `Source: [filename], page [n]` citation.
