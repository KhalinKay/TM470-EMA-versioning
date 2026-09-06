import tempfile
from pathlib import Path

import pytest

from src.rag_assistant.ingestion import load_document, load_documents, split_documents


def test_load_txt_document():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.txt"
        path.write_text("Paragraph one.\n\nParagraph two with more content.", encoding="utf-8")
        docs = load_document(str(path))
        assert len(docs) == 1
        assert "Paragraph one." in docs[0].page_content


def test_load_document_rejects_unsupported_extension():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "notes.md"
        path.write_text("# heading", encoding="utf-8")
        with pytest.raises(ValueError):
            load_document(str(path))


def test_load_documents_combines_multiple_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = Path(tmp_dir) / "a.txt"
        path_b = Path(tmp_dir) / "b.txt"
        path_a.write_text("Content A", encoding="utf-8")
        path_b.write_text("Content B", encoding="utf-8")
        docs = load_documents([str(path_a), str(path_b)])
        assert len(docs) == 2


def test_split_documents_respects_chunk_size():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "long.txt"
        path.write_text("word " * 500, encoding="utf-8")
        docs = load_document(str(path))
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 120 for chunk in chunks)
