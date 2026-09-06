from src.rag_assistant.health import check_ollama


def test_check_ollama_reports_friendly_message_when_unreachable():
    message = check_ollama("http://127.0.0.1:1", timeout=0.5)
    assert message is not None
    assert "Cannot reach the local Ollama server" in message
