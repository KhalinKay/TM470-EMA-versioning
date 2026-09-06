"""System prompt and citation formatting shared by the pipeline and evaluation harness."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a study assistant answering questions using only the document excerpts provided below.

Rules:
1. Answer only using information found in the context. Do not use outside knowledge, even if you know the answer.
2. If the context does not contain the answer, respond that the question falls outside the scope of the uploaded study material, and do not guess, speculate, or fill the gap with general knowledge. This applies even to follow-up questions that are related to the topic but go into detail the material does not cover.
3. {length_instruction}
4. Do not fabricate citations; only refer to the excerpts given.

Context:
{context}

Conversation history:
{history}

Question: {question}
Answer:"""

CONCISE_INSTRUCTION = "Be concise and directly address the question in a short paragraph."
DETAILED_INSTRUCTION = (
    "Answer thoroughly: cover the relevant background, reasoning and any nuance "
    "the context supports, using multiple sentences or paragraphs rather than a "
    "single short line."
)
ANSWER_STYLES = {"concise": CONCISE_INSTRUCTION, "detailed": DETAILED_INSTRUCTION}
DEFAULT_ANSWER_STYLE = "concise"


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


def format_excerpt(doc: Document, max_chars: int = 320) -> str:
    """Render a retrieved chunk as a labelled excerpt for a 'view sources' panel."""
    text = doc.page_content.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    header = f"{_filename(doc)}, page {_page_label(doc)}"
    score = doc.metadata.get("score")
    if isinstance(score, (int, float)):
        header += f" (distance {score:.3f}, lower is closer)"
    return f"{header}\n{text}"
