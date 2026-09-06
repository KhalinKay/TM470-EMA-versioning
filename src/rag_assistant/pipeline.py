"""End-to-end RAG pipeline: ingest documents, retrieve chunks, generate cited answers."""
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import Settings, SETTINGS
from .ingestion import load_documents, split_documents
from .indexing import get_embeddings, build_index, save_index, load_index
from .memory import ConversationMemory
from .prompts import SYSTEM_PROMPT, format_context, format_citation


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for the study assistant.

    `embeddings` and `llm` can be injected (e.g. in tests) to avoid requiring
    a live Ollama server; by default they connect to the configured local server.
    """

    def __init__(self, settings: Settings = SETTINGS, embeddings=None, llm=None):
        self.settings = settings
        self.embeddings = embeddings or get_embeddings(settings.embedding_model, settings.ollama_base_url)
        self.llm = llm or ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0.0)
        self.index = None
        self.memory = ConversationMemory(
            max_turns=settings.memory_max_turns,
            max_tokens=settings.memory_max_tokens,
        )

    def ingest(self, file_paths: List[str]) -> int:
        """Load, chunk and add the given documents to the index. Returns the newly added chunk count.

        Adds to any existing index rather than replacing it, so uploads and the sample
        pack can be combined in the same session.
        """
        documents = load_documents(file_paths)
        chunks = split_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        if self.index is None:
            self.index = build_index(chunks, self.embeddings)
        else:
            self.index.add_documents(chunks)
        return len(chunks)

    @property
    def total_chunks(self) -> int:
        """Total chunks currently in the index, across all ingested documents."""
        return self.index.index.ntotal if self.index is not None else 0

    def clear_documents(self) -> None:
        """Discard the index and conversation memory, returning to a blank session."""
        self.index = None
        self.memory.clear()

    def save(self, path: Optional[str] = None) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Call ingest() first.")
        save_index(self.index, path or self.settings.index_dir)

    def load(self, path: Optional[str] = None) -> None:
        self.index = load_index(path or self.settings.index_dir, self.embeddings)

    def retrieve(self, question: str) -> List[Document]:
        if self.index is None:
            raise RuntimeError("No index loaded. Ingest or load documents first.")
        return self.index.similarity_search(question, k=self.settings.top_k)

    def generate(self, question: str, docs: List[Document]) -> str:
        context = format_context(docs)
        history = self.memory.as_text()
        prompt = SYSTEM_PROMPT.format(context=context, history=history, question=question)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content).strip()

    def _citations_block(self, docs: List[Document]) -> str:
        seen = []
        for doc in docs:
            citation = format_citation(doc)
            if citation not in seen:
                seen.append(citation)
        return "\n".join(seen)

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Retrieve relevant chunks, generate a grounded answer, and append citations."""
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        citations = self._citations_block(docs)
        full_answer = f"{answer}\n\n{citations}" if citations else answer
        self.memory.add_turn(question, full_answer)
        return full_answer, docs

    def reset_memory(self) -> None:
        self.memory.clear()
