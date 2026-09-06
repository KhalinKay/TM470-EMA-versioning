"""Document loading and chunking.

Supports PDF, plain text and Word (.docx) documents, matching the three
formats the RAG Study Assistant ingests. Chunking uses
RecursiveCharacterTextSplitter, which the TruLens chunk-size evaluation
confirmed performs best at 512 characters / 64 overlap.
"""
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(file_path: str) -> List[Document]:
    """Load a single PDF, .txt or .docx file into LangChain Documents."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return loader.load()


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load multiple documents, preserving per-file source metadata."""
    documents: List[Document] = []
    for path in file_paths:
        documents.extend(load_document(path))
    return documents


def load_documents_tolerant(file_paths: List[str]) -> Tuple[List[Document], List[Tuple[str, str]]]:
    """Load multiple documents, skipping any that fail rather than aborting the
    whole batch. Returns the successfully loaded documents alongside a
    (filename, error message) pair for each file that could not be loaded, so a
    single unsupported or corrupted file in a multi-file upload does not prevent
    the rest of the batch from being ingested.
    """
    documents: List[Document] = []
    failures: List[Tuple[str, str]] = []
    for path in file_paths:
        try:
            documents.extend(load_document(path))
        except Exception as exc:  # noqa: BLE001 - any loader failure should be reported, not just ValueError
            failures.append((Path(path).name, str(exc)))
    return documents, failures


def split_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """Split documents into overlapping chunks along paragraph/sentence/word boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
