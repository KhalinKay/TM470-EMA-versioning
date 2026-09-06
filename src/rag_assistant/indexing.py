"""Embedding and FAISS vector index management."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def get_embeddings(model: str, base_url: str) -> OllamaEmbeddings:
    """Return an embeddings client for a locally running Ollama server."""
    return OllamaEmbeddings(model=model, base_url=base_url)


def build_index(chunks: List[Document], embeddings) -> FAISS:
    """Embed document chunks and build an in-memory FAISS index."""
    return FAISS.from_documents(chunks, embeddings)


def save_index(index: FAISS, path: str) -> None:
    """Persist a FAISS index to disk so it can be reloaded without re-embedding."""
    Path(path).mkdir(parents=True, exist_ok=True)
    index.save_local(path)


def load_index(path: str, embeddings) -> FAISS:
    """Load a previously saved FAISS index.

    allow_dangerous_deserialization is safe here: the index is only ever
    written by this application to a local, user-controlled directory.
    """
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def index_exists(path: str) -> bool:
    return (Path(path) / "index.faiss").exists()
