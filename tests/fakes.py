"""Lightweight fakes for Ollama embeddings/LLM so tests run fully offline."""
import hashlib
from typing import List

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic, dependency-free stand-in for OllamaEmbeddings."""

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[:16]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stand-in for ChatOllama that echoes back a canned answer."""

    def __init__(self, answer: str = "This is a test answer."):
        self.answer = answer
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return _FakeResponse(self.answer)
