"""End-to-end RAG pipeline: ingest documents, retrieve chunks, generate cited answers."""
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .config import Settings, SETTINGS
from .ingestion import load_documents_tolerant, split_documents
from .indexing import get_embeddings, build_index, save_index, load_index
from .memory import ConversationMemory
from .prompts import SYSTEM_PROMPT, ANSWER_STYLES, CONCISE_INSTRUCTION, format_context, format_citation


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
        # Maps a source filename to the FAISS chunk ids it contributed, so a single
        # document can be removed from the index without rebuilding it from scratch.
        self.doc_chunk_ids: Dict[str, List[str]] = {}
        # Filenames replaced by the most recent ingest() call (same name re-uploaded),
        # so the UI can tell a user their older version was swapped out rather than
        # silently duplicated in the index.
        self.last_replaced_documents: List[str] = []
        # (filename, error message) pairs for files the most recent ingest() call
        # could not load, so one bad file in a batch does not silently drop the rest.
        self.last_failed_documents: List[Tuple[str, str]] = []

    def set_llm_model(self, model_name: str) -> None:
        """Switch the generation model for subsequent questions.

        Only affects generation: the embedding model stays fixed, since the FAISS
        index was built with it and switching it would require re-embedding every
        loaded document.
        """
        self.settings.llm_model = model_name
        self.llm = ChatOllama(model=model_name, base_url=self.settings.ollama_base_url, temperature=0.0)

    def ingest(self, file_paths: List[str]) -> int:
        """Load, chunk and add the given documents to the index. Returns the newly added chunk count.

        Adds to any existing index rather than replacing it, so uploads and the sample
        pack can be combined in the same session. If a filename being ingested matches
        one already loaded, its previous chunks are removed first, so re-uploading an
        edited version of a document replaces the old copy instead of duplicating it
        alongside the new one. This also covers dropping a same-named file into an
        upload widget that still holds the old one, since Gradio resends every file
        the widget has ever held, not just the newest one: only the last path for a
        given filename in this call is kept. Replaced filenames are recorded in
        `last_replaced_documents` for the caller to report to the user.

        A file that fails to load (an unsupported type or a corrupted document) is
        skipped rather than aborting the whole batch, so the rest of a multi-file
        upload still succeeds; skipped filenames and their errors are recorded in
        `last_failed_documents`.
        """
        deduped_paths: Dict[str, str] = {}
        for path in file_paths:
            deduped_paths[Path(path).name] = path
        documents, failures = load_documents_tolerant(list(deduped_paths.values()))
        self.last_failed_documents = failures
        chunks = split_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        self.last_replaced_documents = []
        if not chunks:
            return 0
        incoming_names = {Path(chunk.metadata.get("source", "unknown")).name for chunk in chunks}
        replaced = sorted(incoming_names & self.doc_chunk_ids.keys())
        for filename in replaced:
            self._discard_chunks(filename)
        if self.index is None:
            self.index = build_index(chunks, self.embeddings)
            ids = [self.index.index_to_docstore_id[i] for i in range(len(chunks))]
        else:
            ids = self.index.add_documents(chunks)
        self._track_chunk_ids(chunks, ids)
        self.last_replaced_documents = replaced
        return len(chunks)

    def _track_chunk_ids(self, chunks: List[Document], ids: List[str]) -> None:
        for chunk, chunk_id in zip(chunks, ids):
            source = Path(chunk.metadata.get("source", "unknown")).name
            self.doc_chunk_ids.setdefault(source, []).append(chunk_id)

    @property
    def total_chunks(self) -> int:
        """Total chunks currently in the index, across all ingested documents."""
        return self.index.index.ntotal if self.index is not None else 0

    @property
    def loaded_documents(self) -> List[str]:
        """Filenames currently contributing chunks to the index, sorted for display."""
        return sorted(self.doc_chunk_ids.keys())

    def remove_document(self, filename: str) -> None:
        """Remove a single previously ingested document's chunks from the index.

        Leaves other documents and the conversation memory untouched. If the
        removed document was the last one in the index, the index is discarded
        entirely so subsequent queries correctly report that nothing is loaded.
        """
        self._discard_chunks(filename)

    def _discard_chunks(self, filename: str) -> None:
        """Remove a filename's chunks from the index and stop tracking it, without
        touching the index at all if that filename was never loaded."""
        ids = self.doc_chunk_ids.pop(filename, None)
        if not ids or self.index is None:
            return
        self.index.delete(ids)
        if self.total_chunks == 0:
            self.index = None

    def clear_documents(self) -> None:
        """Discard the index and conversation memory, returning to a blank session."""
        self.index = None
        self.memory.clear()
        self.doc_chunk_ids = {}

    def save(self, path: Optional[str] = None) -> None:
        """Persist the index and the filename-to-chunk-id manifest so a session can be resumed."""
        if self.index is None:
            raise RuntimeError("No index to save. Call ingest() first.")
        target = path or self.settings.index_dir
        save_index(self.index, target)
        manifest_path = Path(target) / "doc_chunk_ids.json"
        manifest_path.write_text(json.dumps(self.doc_chunk_ids), encoding="utf-8")

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved index and its document manifest, replacing the current session."""
        target = path or self.settings.index_dir
        self.index = load_index(target, self.embeddings)
        manifest_path = Path(target) / "doc_chunk_ids.json"
        if manifest_path.exists():
            self.doc_chunk_ids = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            self.doc_chunk_ids = {}

    def retrieve(self, question: str, scope: Optional[List[str]] = None) -> List[Document]:
        """Retrieve the top_k most relevant chunks, optionally restricted to a subset
        of loaded documents (matched by filename, as tracked in doc_chunk_ids).

        Each returned Document carries the raw FAISS distance score under
        metadata["score"] (lower means closer/more relevant), copied onto a new
        Document rather than mutated in place so the underlying FAISS docstore
        entries are never modified.
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Ingest or load documents first.")
        filter_fn = None
        if scope:
            allowed = set(scope)

            def filter_fn(metadata: dict) -> bool:
                return Path(metadata.get("source", "")).name in allowed

        pairs = self.index.similarity_search_with_score(question, k=self.settings.top_k, filter=filter_fn)
        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "score": float(score)})
            for doc, score in pairs
        ]

    def generate(self, question: str, docs: List[Document]) -> str:
        context = format_context(docs)
        history = self.memory.as_text()
        length_instruction = ANSWER_STYLES.get(self.settings.answer_style, CONCISE_INSTRUCTION)
        prompt = SYSTEM_PROMPT.format(
            context=context, history=history, question=question, length_instruction=length_instruction
        )
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content).strip()

    def generate_stream(self, question: str, docs: List[Document]) -> Iterator[str]:
        """Yield the answer incrementally as the model produces it.

        Falls back to a single yield of the complete answer if the underlying
        LLM (e.g. a test fake) does not support streaming.
        """
        context = format_context(docs)
        history = self.memory.as_text()
        length_instruction = ANSWER_STYLES.get(self.settings.answer_style, CONCISE_INSTRUCTION)
        prompt = SYSTEM_PROMPT.format(
            context=context, history=history, question=question, length_instruction=length_instruction
        )
        if not hasattr(self.llm, "stream"):
            yield self.generate(question, docs)
            return
        for chunk in self.llm.stream(prompt):
            content = getattr(chunk, "content", chunk)
            if content:
                yield str(content)

    def _citations_block(self, docs: List[Document]) -> str:
        seen = []
        for doc in docs:
            citation = format_citation(doc)
            if citation not in seen:
                seen.append(citation)
        return "\n".join(seen)

    def query(self, question: str, scope: Optional[List[str]] = None) -> Tuple[str, List[Document]]:
        """Retrieve relevant chunks, generate a grounded answer, and append citations."""
        docs = self.retrieve(question, scope=scope)
        answer = self.generate(question, docs)
        citations = self._citations_block(docs)
        full_answer = f"{answer}\n\n{citations}" if citations else answer
        self.memory.add_turn(question, full_answer)
        return full_answer, docs

    def query_stream(self, question: str, scope: Optional[List[str]] = None) -> Iterator[Tuple[str, List[Document]]]:
        """Retrieve context once, then yield the growing answer as it streams in.

        Conversation memory is only updated once generation is complete, on the
        final yield, so a mid-stream read never sees a partial answer recorded
        as history.
        """
        docs = self.retrieve(question, scope=scope)
        answer_so_far = ""
        for piece in self.generate_stream(question, docs):
            answer_so_far += piece
            yield answer_so_far, docs
        citations = self._citations_block(docs)
        full_answer = f"{answer_so_far.strip()}\n\n{citations}" if citations else answer_so_far.strip()
        self.memory.add_turn(question, full_answer)
        yield full_answer, docs

    def reset_memory(self) -> None:
        self.memory.clear()
