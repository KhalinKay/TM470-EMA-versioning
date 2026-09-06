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


def test_remove_document_drops_only_that_documents_chunks():
    pipeline = _pipeline()
    with tempfile.TemporaryDirectory() as tmp_dir:
        notes_path = Path(tmp_dir) / "notes.txt"
        notes_path.write_text("RAG combines retrieval with generation. " * 20, encoding="utf-8")
        other_path = Path(tmp_dir) / "other.txt"
        other_path.write_text("FAISS is a vector similarity search library. " * 20, encoding="utf-8")
        pipeline.ingest([str(notes_path)])
        pipeline.ingest([str(other_path)])

    assert pipeline.loaded_documents == ["notes.txt", "other.txt"]
    total_before = pipeline.total_chunks

    pipeline.remove_document("notes.txt")

    assert pipeline.loaded_documents == ["other.txt"]
    assert pipeline.total_chunks < total_before
    assert pipeline.index is not None


def test_remove_last_document_discards_index():
    pipeline = _pipeline()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.txt"
        path.write_text("RAG combines retrieval with generation. " * 20, encoding="utf-8")
        pipeline.ingest([str(path)])

    pipeline.remove_document("notes.txt")

    assert pipeline.index is None
    assert pipeline.loaded_documents == []


def test_save_and_load_round_trip_preserves_documents():
    pipeline = _pipeline()
    with tempfile.TemporaryDirectory() as tmp_dir:
        docs_dir = Path(tmp_dir) / "docs"
        docs_dir.mkdir()
        path = docs_dir / "notes.txt"
        path.write_text("RAG combines retrieval with generation. " * 20, encoding="utf-8")
        pipeline.ingest([str(path)])

        index_dir = Path(tmp_dir) / "index"
        pipeline.save(str(index_dir))

        reloaded = _pipeline()
        reloaded.load(str(index_dir))

        assert reloaded.loaded_documents == ["notes.txt"]
        assert reloaded.total_chunks == pipeline.total_chunks
        answer, docs = reloaded.query("What is RAG?")
        assert "Source: notes.txt" in answer
        assert len(docs) > 0


def test_query_stream_yields_growing_answer_and_updates_memory():
    pipeline = _pipeline(answer="This is a streamed answer.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.txt"
        path.write_text("RAG combines retrieval with generation. " * 20, encoding="utf-8")
        pipeline.ingest([str(path)])

    partials = [answer for answer, _ in pipeline.query_stream("What is RAG?")]

    assert len(partials) > 1
    assert partials[-2] == "This is a streamed answer."
    assert "Source: notes.txt" in partials[-1]
    assert len(pipeline.memory.turns) == 1
    assert "Source: notes.txt" in pipeline.memory.turns[0][1]
