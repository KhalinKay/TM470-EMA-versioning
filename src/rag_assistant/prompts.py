"""System prompt and citation formatting shared by the pipeline and evaluation harness."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a study assistant answering questions using only the document excerpts provided below.

Rules:
1. Answer only using information found in the context. Do not use outside knowledge, even if you know the answer.
2. If the context does not contain the answer, respond that the question falls outside the scope of the uploaded study material, and do not guess, speculate, or fill the gap with general knowledge. This applies even to follow-up questions that are related to the topic but go into detail the material does not cover.
3. Be concise and directly address the question.
4. Do not fabricate citations; only refer to the excerpts given.

Context:
{context}

Conversation history:
{history}

Question: {question}
Answer:"""


def _filename(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    return Path(source).name if source != "unknown" else "unknown"


def _page_label(doc: Document) -> str:
    page = doc.metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "n/a"


def format_context(docs: List[Document]) -> str:
    """Render retrieved chunks with inline source/page labels for the prompt."""
    parts = []
    for doc in docs:
        parts.append(f"[{_filename(doc)}, page {_page_label(doc)}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_citation(doc: Document) -> str:
    """Render the mandatory 'Source: [filename], page [n]' citation line."""
    return f"Source: {_filename(doc)}, page {_page_label(doc)}"
