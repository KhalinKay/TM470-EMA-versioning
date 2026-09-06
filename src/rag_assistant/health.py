"""Connectivity check for the local Ollama server.

Used at startup and before ingestion so a missing or unreachable Ollama
server produces one clear, actionable message instead of a raw connection
exception surfacing from deep inside the embeddings/LLM client.
"""
import json
from typing import List, Optional
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


def list_ollama_models(base_url: str, timeout: float = 2.0) -> List[str]:
    """Return the names of models currently pulled in the local Ollama server.

    Returns an empty list if the server is unreachable or the response cannot
    be parsed, rather than raising, since this is only used to populate an
    optional dropdown.
    """
    try:
        with request.urlopen(f"{base_url}/api/tags", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return sorted(model["name"] for model in payload.get("models", []))
    except (error.URLError, OSError, ValueError, KeyError):
        return []

