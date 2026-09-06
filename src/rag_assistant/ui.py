"""Gradio web interface: file upload panel plus a chat window.

The FAISS index and conversation memory are kept inside a per-session
RAGPipeline held in gr.State, so re-querying does not rebuild the index.
"""
from pathlib import Path
from typing import List, Optional

import gradio as gr

from .config import SETTINGS
from .pipeline import RAGPipeline

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


def handle_upload(files, pipeline: Optional[RAGPipeline]):
    if not files:
        return "No files selected.", pipeline
    pipeline = pipeline or _new_pipeline()
    try:
        num_chunks = pipeline.ingest(_file_paths(files))
    except Exception as exc:  # surfaced to the user rather than crashing the UI
        return f"Failed to process documents: {exc}", pipeline
    return (
        f"Added {len(files)} document(s) ({num_chunks} new chunks, {pipeline.total_chunks} total in the index). "
        "Ask a question below.",
        pipeline,
    )


def handle_load_samples(pipeline: Optional[RAGPipeline]):
    paths = [str(p) for p in SAMPLE_DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"}]
    if not paths:
        return f"No sample documents found in {SAMPLE_DOCS_DIR}.", pipeline
    pipeline = pipeline or _new_pipeline()
    try:
        num_chunks = pipeline.ingest(paths)
    except Exception as exc:
        return f"Failed to load sample documents: {exc}", pipeline
    names = ", ".join(Path(p).name for p in paths)
    return (
        f"Loaded sample documents ({names}): {num_chunks} new chunks, {pipeline.total_chunks} total in the index. "
        "Try one of the example questions below, or ask your own.",
        pipeline,
    )


def handle_remove_documents(pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.clear_documents()
    return "All documents removed. Upload files or load the sample pack to begin.", pipeline, []


def handle_message(message: str, history, pipeline: Optional[RAGPipeline]):
    """Gradio's Chatbot (v6) uses the OpenAI-style messages format: a list of
    {"role": "user"|"assistant", "content": ...} dicts rather than tuples."""
    history = history or []
    if not message:
        return history, pipeline, ""
    if pipeline is None or pipeline.index is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload one or more documents first."},
        ]
        return history, pipeline, ""
    try:
        answer, _ = pipeline.query(message)
    except Exception as exc:
        answer = f"Error while generating a response: {exc}"
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, pipeline, ""


def handle_clear(pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.reset_memory()
    return [], pipeline


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
                remove_docs_btn = gr.Button("Remove all documents", size="sm")
            upload_status = gr.Textbox(label="Index status", interactive=False)

        chatbot = gr.Chatbot(label="Conversation", height=450)
        question_box = gr.Textbox(label="Ask a question", placeholder="e.g. What are the Five Pillars of Islam?")
        gr.Examples(examples=SAMPLE_QUESTIONS, inputs=question_box, label="Example questions (after loading the sample documents)")
        with gr.Row():
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear conversation")

        file_upload.upload(
            handle_upload,
            inputs=[file_upload, pipeline_state],
            outputs=[upload_status, pipeline_state],
        )
        sample_btn.click(
            handle_load_samples,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state],
        )
        remove_docs_btn.click(
            handle_remove_documents,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, chatbot],
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

    return demo

