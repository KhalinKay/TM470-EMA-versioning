import tempfile
from pathlib import Path

from src.rag_assistant.ingestion import load_document, split_documents
from src.rag_assistant.indexing import build_index, save_index, load_index, index_exists
from tests.fakes import FakeEmbeddings


def _sample_chunks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.txt"
        path.write_text(
            "Retrieval-Augmented Generation grounds language models in retrieved text. " * 10,
            encoding="utf-8",
        )
        docs = load_document(str(path))
        return split_documents(docs, chunk_size=100, chunk_overlap=20)


def test_build_index_returns_searchable_faiss_store():
    chunks = _sample_chunks()
    index = build_index(chunks, FakeEmbeddings())
    results = index.similarity_search("Retrieval-Augmented Generation", k=2)
    assert len(results) == 2


def test_save_and_load_index_round_trip():
    chunks = _sample_chunks()
    embeddings = FakeEmbeddings()
    index = build_index(chunks, embeddings)
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert not index_exists(tmp_dir)
        save_index(index, tmp_dir)
        assert index_exists(tmp_dir)
        reloaded = load_index(tmp_dir, embeddings)
        results = reloaded.similarity_search("Retrieval-Augmented Generation", k=1)
        assert len(results) == 1
