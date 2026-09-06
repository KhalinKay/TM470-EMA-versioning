import tempfile
from pathlib import Path

from src.rag_assistant.config import Settings
from src.rag_assistant.pipeline import RAGPipeline
from tests.fakes import FakeEmbeddings, FakeLLM


def _pipeline(answer: str = "The documents describe a RAG pipeline.") -> RAGPipeline:
    settings = Settings(chunk_size=100, chunk_overlap=20, top_k=2)
    return RAGPipeline(settings=settings, embeddings=FakeEmbeddings(), llm=FakeLLM(answer))


def test_query_returns_answer_with_citation():
    pipeline = _pipeline()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.txt"
        path.write_text("RAG combines retrieval with generation. " * 20, encoding="utf-8")
        pipeline.ingest([str(path)])

    answer, docs = pipeline.query("What is RAG?")
    assert "The documents describe a RAG pipeline." in answer
    assert "Source: notes.txt" in answer
    assert len(docs) > 0


def test_query_before_ingest_raises():
    pipeline = _pipeline()
    try:
        pipeline.query("What is RAG?")
        assert False, "Expected a RuntimeError when no index has been loaded."
    except RuntimeError:
        pass


def test_memory_accumulates_and_resets():
    pipeline = _pipeline()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.txt"
        path.write_text("RAG combines retrieval with generation. " * 20, encoding="utf-8")
        pipeline.ingest([str(path)])

    pipeline.query("What is RAG?")
    assert len(pipeline.memory.turns) == 1
    pipeline.reset_memory()
    assert len(pipeline.memory.turns) == 0
