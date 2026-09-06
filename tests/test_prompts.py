from langchain_core.documents import Document

from src.rag_assistant.prompts import format_citation, format_excerpt


def test_format_citation_uses_filename_and_one_based_page():
    doc = Document(page_content="text", metadata={"source": "/tmp/notes.pdf", "page": 2})
    assert format_citation(doc) == "Source: notes.pdf, page 3"


def test_format_citation_falls_back_to_not_available_for_missing_page():
    doc = Document(page_content="text", metadata={"source": "/tmp/notes.txt"})
    assert format_citation(doc) == "Source: notes.txt, page n/a"


def test_format_excerpt_truncates_long_content():
    doc = Document(page_content="x" * 500, metadata={"source": "notes.txt"})
    excerpt = format_excerpt(doc, max_chars=320)
    assert excerpt.startswith("notes.txt, page n/a\n")
    assert excerpt.endswith("...")
    assert len(excerpt) < 500


def test_format_excerpt_leaves_short_content_untouched():
    doc = Document(page_content="short text", metadata={"source": "notes.txt"})
    excerpt = format_excerpt(doc)
    assert excerpt == "notes.txt, page n/a\nshort text"
