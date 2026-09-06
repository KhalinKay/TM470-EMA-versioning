"""Gradio web interface: file upload panel plus a chat window.

The FAISS index and conversation memory are kept inside a per-session
RAGPipeline held in gr.State, so re-querying does not rebuild the index.
"""
import tempfile
from pathlib import Path
from typing import List, Optional

import gradio as gr

from .config import SETTINGS
from .health import check_ollama
from .indexing import index_exists
from .pipeline import RAGPipeline
from .prompts import format_excerpt

# Shipped example corpus: a short "History and Philosophy of Religion" study pack.
SAMPLE_DOCS_DIR = Path("data/sample_docs")
SAMPLE_QUESTIONS = [
    "What religious reform is Akhenaten associated with in ancient Egypt?",
    "What is the significance of the Council of Nicaea in 325 CE?",
    "What are the Four Noble Truths in Buddhism?",
    "How did the Sunni-Shia split within Islam originate?",
    "How does Confucianism differ from Daoism in its approach to social order?",
]

# Muted teal/amber/stone palette: deliberately not the default indigo/violet "AI" look.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
)


def _new_pipeline() -> RAGPipeline:
    return RAGPipeline(SETTINGS)


def _file_paths(files) -> List[str]:
    return [f.name if hasattr(f, "name") else str(f) for f in files]


def _doc_choices(pipeline: Optional[RAGPipeline]):
    return gr.update(choices=pipeline.loaded_documents if pipeline else [], value=[])


def handle_upload(files, pipeline: Optional[RAGPipeline]):
    if not files:
        return "No files selected.", pipeline, _doc_choices(pipeline)
    connection_error = check_ollama(SETTINGS.ollama_base_url)
    if connection_error:
        return connection_error, pipeline, _doc_choices(pipeline)
    pipeline = pipeline or _new_pipeline()
    try:
        num_chunks = pipeline.ingest(_file_paths(files))
    except Exception as exc:  # surfaced to the user rather than crashing the UI
        return f"Failed to process documents: {exc}", pipeline, _doc_choices(pipeline)
    return (
        f"Added {len(files)} document(s) ({num_chunks} new chunks, {pipeline.total_chunks} total in the index). "
        "Ask a question below.",
        pipeline,
        _doc_choices(pipeline),
    )


def handle_load_samples(pipeline: Optional[RAGPipeline]):
    paths = [str(p) for p in SAMPLE_DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"}]
    if not paths:
        return f"No sample documents found in {SAMPLE_DOCS_DIR}.", pipeline, _doc_choices(pipeline)
    connection_error = check_ollama(SETTINGS.ollama_base_url)
    if connection_error:
        return connection_error, pipeline, _doc_choices(pipeline)
    pipeline = pipeline or _new_pipeline()
    try:
        num_chunks = pipeline.ingest(paths)
    except Exception as exc:
        return f"Failed to load sample documents: {exc}", pipeline, _doc_choices(pipeline)
    names = ", ".join(Path(p).name for p in paths)
    return (
        f"Loaded sample documents ({names}): {num_chunks} new chunks, {pipeline.total_chunks} total in the index. "
        "Try one of the example questions below, or ask your own.",
        pipeline,
        _doc_choices(pipeline),
    )


def handle_remove_documents(pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.clear_documents()
    return "All documents removed. Upload files or load the sample pack to begin.", pipeline, [], _doc_choices(pipeline)


def handle_remove_selected(selected: List[str], pipeline: Optional[RAGPipeline]):
    if not selected or pipeline is None:
        return "No documents selected to remove.", pipeline, _doc_choices(pipeline)
    for filename in selected:
        pipeline.remove_document(filename)
    if pipeline.index is None:
        status = "Removed the selected document(s). No documents remain in the index."
    else:
        status = (
            f"Removed {len(selected)} document(s). {pipeline.total_chunks} chunk(s) remain across "
            f"{len(pipeline.loaded_documents)} document(s)."
        )
    return status, pipeline, _doc_choices(pipeline)


def handle_save_session(pipeline: Optional[RAGPipeline]):
    if pipeline is None or pipeline.index is None:
        return "No documents loaded, so there is nothing to save."
    try:
        pipeline.save()
    except Exception as exc:
        return f"Failed to save the session: {exc}"
    return (
        f"Session saved to disk ({pipeline.total_chunks} chunks, {len(pipeline.loaded_documents)} document(s)). "
        "Use 'Resume last session' next time to reload it without re-embedding."
    )


def handle_resume_session(pipeline: Optional[RAGPipeline]):
    if not index_exists(SETTINGS.index_dir):
        return "No saved session found on disk.", pipeline, _doc_choices(pipeline)
    pipeline = pipeline or _new_pipeline()
    try:
        pipeline.load()
    except Exception as exc:
        return f"Failed to resume the saved session: {exc}", pipeline, _doc_choices(pipeline)
    return (
        f"Resumed the saved session: {len(pipeline.loaded_documents)} document(s), "
        f"{pipeline.total_chunks} chunks. Ask a question below.",
        pipeline,
        _doc_choices(pipeline),
    )


def handle_top_k_change(value, pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.settings.top_k = int(value)
    return pipeline


def handle_connection_check():
    """Re-checked on every page load, not just once at process startup, so the
    banner reflects whether Ollama is reachable right now."""
    connection_error = check_ollama(SETTINGS.ollama_base_url)
    if connection_error:
        return gr.update(value=f"**{connection_error}**", visible=True)
    return gr.update(visible=False)


def handle_message(message: str, history, pipeline: Optional[RAGPipeline]):
    """Gradio's Chatbot (v6) uses the OpenAI-style messages format: a list of
    {"role": "user"|"assistant", "content": ...} dicts rather than tuples.

    This is a generator so the answer streams into the chat window token by
    token instead of appearing all at once, and finishes with a collapsible
    'View retrieved excerpts' message so the grounding for the answer is
    inspectable without leaving the chat.
    """
    history = history or []
    if not message:
        yield history, pipeline, ""
        return
    if pipeline is None or pipeline.index is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload one or more documents first."},
        ]
        yield history, pipeline, ""
        return

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield history, pipeline, ""

    last_docs = []
    try:
        for partial_answer, docs in pipeline.query_stream(message):
            last_docs = docs
            history[-1] = {"role": "assistant", "content": partial_answer}
            yield history, pipeline, ""
    except Exception as exc:
        history[-1] = {"role": "assistant", "content": f"Error while generating a response: {exc}"}
        yield history, pipeline, ""
        return

    if last_docs:
        excerpts = "\n\n".join(format_excerpt(doc) for doc in last_docs)
        history = history + [
            {"role": "assistant", "content": excerpts, "metadata": {"title": "View retrieved excerpts"}}
        ]
        yield history, pipeline, ""


def handle_clear(pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.reset_memory()
    return [], pipeline


def handle_download_transcript(history):
    if not history:
        return None
    lines = []
    for message in history:
        role = "You" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    text = "\n\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG Study Assistant") as demo:
        gr.Markdown(
            "# RAG Study Assistant\n"
            "Upload your own PDF, .txt or .docx study documents, or click **Load sample documents** to try a "
            "bundled History and Philosophy of Religion study pack. Everything runs locally through Ollama; "
            "no data leaves this machine.\n\n"
            "**Answers stay on topic by design.** Every answer is grounded in your uploaded material and cites "
            "its source; a question outside that material is declined rather than guessed at. This is what "
            "keeps the assistant trustworthy for study and revision, not a limitation to work around. See "
            "below for why."
        )
        connection_banner = gr.Markdown(visible=False)
        with gr.Accordion("Why does it decline some questions?", open=False):
            gr.Markdown(
                "This assistant only answers from the documents you have loaded, so every answer can be traced "
                "back to a source. If you ask something the material does not cover, even a closely related "
                "follow-up, it will tell you the question is out of scope instead of filling the gap with the "
                "model's own general knowledge. That guarantee is what makes the citations meaningful: an answer "
                "is either backed by your material or it is declined, never guessed. To widen what it can "
                "answer, upload documents that cover the topic you want to ask about."
            )
        pipeline_state = gr.State(None)

        with gr.Row():
            file_upload = gr.Files(label="Upload study documents", file_types=[".pdf", ".txt", ".docx"])
            with gr.Column():
                sample_btn = gr.Button("Load sample documents (History of Religion)")
                with gr.Row():
                    save_session_btn = gr.Button("Save session", size="sm")
                    resume_session_btn = gr.Button("Resume last session", size="sm")
                remove_docs_btn = gr.Button("Remove all documents", size="sm")
            upload_status = gr.Textbox(label="Index status", interactive=False)

        with gr.Accordion("Loaded documents", open=False):
            doc_list = gr.CheckboxGroup(label="Select document(s) to remove individually", choices=[])
            remove_selected_btn = gr.Button("Remove selected", size="sm")

        with gr.Accordion("Advanced settings", open=False):
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, step=1, value=SETTINGS.top_k,
                label="Chunks retrieved per question (top_k)",
                info="Higher values give the model more context per question, at the cost of a longer prompt. "
                     "Applies to the next question asked.",
            )

        chatbot = gr.Chatbot(label="Conversation", height=450)
        question_box = gr.Textbox(label="Ask a question", placeholder="e.g. What are the Five Pillars of Islam?")
        gr.Examples(examples=SAMPLE_QUESTIONS, inputs=question_box, label="Example questions (after loading the sample documents)")
        with gr.Row():
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear conversation")
            download_btn = gr.DownloadButton("Download transcript", size="sm")

        file_upload.upload(
            handle_upload,
            inputs=[file_upload, pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list],
        )
        sample_btn.click(
            handle_load_samples,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list],
        )
        remove_docs_btn.click(
            handle_remove_documents,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, chatbot, doc_list],
        )
        remove_selected_btn.click(
            handle_remove_selected,
            inputs=[doc_list, pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list],
        )
        save_session_btn.click(
            handle_save_session,
            inputs=[pipeline_state],
            outputs=[upload_status],
        )
        resume_session_btn.click(
            handle_resume_session,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list],
        )
        top_k_slider.change(
            handle_top_k_change,
            inputs=[top_k_slider, pipeline_state],
            outputs=[pipeline_state],
        )
        submit_btn.click(
            handle_message,
            inputs=[question_box, chatbot, pipeline_state],
            outputs=[chatbot, pipeline_state, question_box],
        )
        question_box.submit(
            handle_message,
            inputs=[question_box, chatbot, pipeline_state],
            outputs=[chatbot, pipeline_state, question_box],
        )
        clear_btn.click(handle_clear, inputs=[pipeline_state], outputs=[chatbot, pipeline_state])
        download_btn.click(handle_download_transcript, inputs=[chatbot], outputs=[download_btn])
        demo.load(handle_connection_check, outputs=[connection_banner])

    return demo


