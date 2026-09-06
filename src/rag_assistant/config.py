"""Central configuration for the RAG Study Assistant.

Values are read from environment variables (see .env.example) so the
application can be reconfigured without touching code, while keeping
sensible defaults matching the evaluated 512/64 chunking configuration.
"""
from dataclasses import dataclass
import os


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    chunk_size: int = _int_env("CHUNK_SIZE", 512)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 64)
    top_k: int = _int_env("TOP_K", 4)

    index_dir: str = os.getenv("INDEX_DIR", "data/faiss_index")

    memory_max_turns: int = _int_env("MEMORY_MAX_TURNS", 4)
    memory_max_tokens: int = _int_env("MEMORY_MAX_TOKENS", 1500)


SETTINGS = Settings()
