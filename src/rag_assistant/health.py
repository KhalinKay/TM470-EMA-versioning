"""Connectivity check for the local Ollama server.

Used at startup and before ingestion so a missing or unreachable Ollama
server produces one clear, actionable message instead of a raw connection
exception surfacing from deep inside the embeddings/LLM client.
"""
from typing import Optional
from urllib import request, error


def check_ollama(base_url: str, timeout: float = 2.0) -> Optional[str]:
    """Return None if Ollama is reachable at base_url, otherwise a user-facing message."""
    try:
        request.urlopen(f"{base_url}/api/tags", timeout=timeout)
        return None
    except (error.URLError, OSError, ValueError):
        return (
            f"Cannot reach the local Ollama server at {base_url}. "
            "Start it (e.g. run 'ollama serve') and try again."
        )
