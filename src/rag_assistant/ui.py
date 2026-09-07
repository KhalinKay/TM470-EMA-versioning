"""Gradio web interface: file upload panel plus a chat window.

The FAISS index and conversation memory are kept inside a per-session
RAGPipeline held in gr.State, so re-querying does not rebuild the index.
"""
import tempfile
from pathlib import Path
from typing import List, Optional

import gradio as gr

from .config import SETTINGS
from .health import check_ollama, list_ollama_models
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

# A warm, paper-and-ink "study" aesthetic: deliberately not the default indigo/violet
# "AI chatbot" look. Colours are a muted teal (primary action), amber (accents) and a
# warm stone/parchment neutral scale. Fonts are local system fonts only (no network
# fetches), matching the project's fully-offline requirement. Light-mode values are
# repeated as the dark-mode values too, so the look is consistent regardless of the
# viewer's OS/browser colour-scheme preference rather than falling back to Gradio's
# default dark palette.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.Font("Segoe UI"), gr.themes.Font("ui-sans-serif"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")),
    font_mono=(gr.themes.Font("Consolas"), gr.themes.Font("ui-monospace"), gr.themes.Font("monospace")),
).set(
    body_background_fill="#FAF6EE",
    body_background_fill_dark="#FAF6EE",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F3EDE0",
    background_fill_secondary_dark="#F3EDE0",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E5DCC6",
    block_border_color_dark="#E5DCC6",
    block_border_width="1px",
    block_radius="14px",
    block_shadow="0 1px 3px rgba(41, 37, 27, 0.06)",
    panel_background_fill="#F3EDE0",
    panel_background_fill_dark="#F3EDE0",
    panel_border_color="#E5DCC6",
    panel_border_color_dark="#E5DCC6",
    border_color_primary="#E5DCC6",
    border_color_primary_dark="#E5DCC6",
    checkbox_background_color="#FFFFFF",
    checkbox_background_color_dark="#FFFFFF",
    checkbox_background_color_selected="#0F6B62",
    checkbox_background_color_selected_dark="#0F6B62",
    checkbox_border_color="#E5DCC6",
    checkbox_border_color_dark="#E5DCC6",
    checkbox_border_color_selected="#0F6B62",
    checkbox_border_color_selected_dark="#0F6B62",
    checkbox_label_background_fill="#FFFFFF",
    checkbox_label_background_fill_dark="#FFFFFF",
    block_label_text_color="#6B6252",
    block_label_text_color_dark="#6B6252",
    block_title_text_color="#2A2620",
    block_title_text_color_dark="#2A2620",
    body_text_color="#2A2620",
    body_text_color_dark="#2A2620",
    body_text_color_subdued="#7A7161",
    body_text_color_subdued_dark="#7A7161",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E5DCC6",
    input_border_color_dark="#E5DCC6",
    button_primary_background_fill="#0F6B62",
    button_primary_background_fill_dark="#0F6B62",
    button_primary_background_fill_hover="#0C554E",
    button_primary_background_fill_hover_dark="#0C554E",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover="#F3EDE0",
    button_secondary_background_fill_hover_dark="#F3EDE0",
    button_secondary_border_color="#0F6B62",
    button_secondary_border_color_dark="#0F6B62",
    button_secondary_text_color="#0F6B62",
    button_secondary_text_color_dark="#0F6B62",
    border_color_accent="#C68A3B",
    border_color_accent_dark="#C68A3B",
    color_accent="#0F6B62",
    color_accent_soft_dark="#F3EDE0",
    link_text_color="#0F6B62",
    link_text_color_dark="#0F6B62",
)

# Header/hero band, sidebar/chat panel styling and a few small refinements CSS can
# reach that theme variables cannot (max page width, the hero banner, the footer
# note). No @import/remote fonts here, in keeping with the fully-offline requirement.
CUSTOM_CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

#hero {
    background: linear-gradient(135deg, #0F6B62 0%, #134A44 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 6px;
}
#hero h1 {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    color: #FFFFFF !important;
    font-size: 1.7rem;
    margin: 0 0 6px 0;
    letter-spacing: 0.2px;
}
#hero p { color: #DCEFEC !important; margin: 0; font-size: 0.95rem; line-height: 1.5; max-width: 680px; }
#hero .badge {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.35);
    color: #EAF6F4 !important;
    font-size: 0.76rem;
    letter-spacing: 0.3px;
}

/* Two-pane app shell: a tinted sidebar panel for document/session controls next to
   a plain chat pane, instead of stacking mismatched-height controls in one row
   (which left odd blank gaps in the previous single-column layout). */
#sidebar {
    background: #F3EDE0;
    border: 1px solid #E5DCC6;
    border-radius: 16px;
    padding: 16px 16px 20px 16px;
}
#main-chat { padding-top: 4px; }
.sidebar-heading {
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-size: 0.72rem !important;
    font-weight: 600;
    color: #7A7161 !important;
    margin: 4px 0 2px 2px !important;
}

footer#footer-note {
    text-align: center;
    color: #9A9280;
    font-size: 0.78rem;
    margin-top: 18px;
    padding-bottom: 8px;
}
#chatbot.block { border-width: 1px !important; border-color: #E5DCC6 !important; }
"""


def _new_pipeline() -> RAGPipeline:
    return RAGPipeline(SETTINGS)


def _file_paths(files) -> List[str]:
    return [f.name if hasattr(f, "name") else str(f) for f in files]


def _doc_choices(pipeline: Optional[RAGPipeline]):
    return gr.update(choices=pipeline.loaded_documents if pipeline else [], value=[])


def _sample_doc_names() -> set:
    return {p.name for p in SAMPLE_DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"}}


def _examples_visibility(pipeline: Optional[RAGPipeline]):
    """Show the shipped example questions only while they still apply: no documents
    loaded yet, or exactly the sample pack. They name specific sample-pack facts
    (Akhenaten, the Council of Nicaea, ...), so once a tutor's own material is
    loaded instead, showing them would invite clicks that are declined as out of
    scope."""
    no_docs = pipeline is None or not pipeline.loaded_documents
    is_sample_set = no_docs or set(pipeline.loaded_documents) == _sample_doc_names()
    return gr.update(visible=is_sample_set)


def _message_text(content) -> str:
    """Chat message content is normally a plain string, but Gradio's Chatbot can
    round-trip it as a list of {"text": ..., "type": "text"} parts. Handle both."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def handle_upload(files, pipeline: Optional[RAGPipeline]):
    if not files:
        return "No files selected.", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    connection_error = check_ollama(SETTINGS.ollama_base_url)
    if connection_error:
        return connection_error, pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    pipeline = pipeline or _new_pipeline()
    paths = _file_paths(files)
    try:
        num_chunks = pipeline.ingest(paths)
    except Exception as exc:  # surfaced to the user rather than crashing the UI
        return f"Failed to process documents: {exc}", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    replaced_note = ""
    if pipeline.last_replaced_documents:
        replaced_note = f" Replaced existing version(s) of: {', '.join(pipeline.last_replaced_documents)}."
    failed_note = ""
    if pipeline.last_failed_documents:
        failed_list = "; ".join(f"{name} ({error})" for name, error in pipeline.last_failed_documents)
        failed_note = f" Could not process: {failed_list}."
    num_documents = len({Path(p).name for p in paths})
    return (
        f"Added {num_documents} document(s) ({num_chunks} new chunks, {pipeline.total_chunks} total in the index)."
        f"{replaced_note}{failed_note} Ask a question below.",
        pipeline,
        _doc_choices(pipeline),
        _examples_visibility(pipeline),
    )


def handle_load_samples(pipeline: Optional[RAGPipeline]):
    paths = [str(p) for p in SAMPLE_DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"}]
    if not paths:
        return f"No sample documents found in {SAMPLE_DOCS_DIR}.", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    connection_error = check_ollama(SETTINGS.ollama_base_url)
    if connection_error:
        return connection_error, pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    pipeline = pipeline or _new_pipeline()
    try:
        num_chunks = pipeline.ingest(paths)
    except Exception as exc:
        return f"Failed to load sample documents: {exc}", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    names = ", ".join(Path(p).name for p in paths)
    replaced_note = ""
    if pipeline.last_replaced_documents:
        replaced_note = f" Replaced existing version(s) of: {', '.join(pipeline.last_replaced_documents)}."
    failed_note = ""
    if pipeline.last_failed_documents:
        failed_list = "; ".join(f"{name} ({error})" for name, error in pipeline.last_failed_documents)
        failed_note = f" Could not process: {failed_list}."
    return (
        f"Loaded sample documents ({names}): {num_chunks} new chunks, {pipeline.total_chunks} total in the index."
        f"{replaced_note}{failed_note} Try one of the example questions below, or ask your own.",
        pipeline,
        _doc_choices(pipeline),
        _examples_visibility(pipeline),
    )


def handle_remove_documents(pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.clear_documents()
    return (
        "All documents removed. Upload files or load the sample pack to begin.",
        pipeline,
        [],
        _doc_choices(pipeline),
        _examples_visibility(pipeline),
    )


def handle_remove_selected(selected: List[str], pipeline: Optional[RAGPipeline]):
    if not selected or pipeline is None:
        return "No documents selected to remove.", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    for filename in selected:
        pipeline.remove_document(filename)
    if pipeline.index is None:
        status = "Removed the selected document(s). No documents remain in the index."
    else:
        status = (
            f"Removed {len(selected)} document(s). {pipeline.total_chunks} chunk(s) remain across "
            f"{len(pipeline.loaded_documents)} document(s)."
        )
    return status, pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)


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
        return "No saved session found on disk.", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    pipeline = pipeline or _new_pipeline()
    try:
        pipeline.load()
    except Exception as exc:
        return f"Failed to resume the saved session: {exc}", pipeline, _doc_choices(pipeline), _examples_visibility(pipeline)
    return (
        f"Resumed the saved session: {len(pipeline.loaded_documents)} document(s), "
        f"{pipeline.total_chunks} chunks. Ask a question below.",
        pipeline,
        _doc_choices(pipeline),
        _examples_visibility(pipeline),
    )


def handle_top_k_change(value, pipeline: Optional[RAGPipeline]):
    if pipeline is not None:
        pipeline.settings.top_k = int(value)
    return pipeline


def handle_answer_style_change(style: str, pipeline: Optional[RAGPipeline]):
    """Switch between a short, direct answer and a fuller, more explanatory one.

    Applies to subsequent questions only; like the model dropdown, the choice is
    remembered on the shared settings when no pipeline exists yet (no documents
    loaded), so it still takes effect once one is created.
    """
    if not style:
        return pipeline
    if pipeline is not None:
        pipeline.settings.answer_style = style
    else:
        SETTINGS.answer_style = style
    return pipeline


def handle_model_choices():
    """Populate the generation model dropdown from whatever is actually pulled in
    the local Ollama server, re-checked on every page load like the connection
    banner, so a model pulled or removed since the app started is reflected.

    The configured embedding model is excluded: it is used to build the FAISS
    index and is not a chat model, so selecting it for generation would fail.
    """
    embedding_base = SETTINGS.embedding_model.split(":")[0]
    models = [m for m in list_ollama_models(SETTINGS.ollama_base_url) if m.split(":")[0] != embedding_base]
    if not models:
        return gr.update(choices=[SETTINGS.llm_model], value=SETTINGS.llm_model)
    default = SETTINGS.llm_model if SETTINGS.llm_model in models else models[0]
    return gr.update(choices=models, value=default)


def handle_model_change(model_name: str, pipeline: Optional[RAGPipeline]):
    if not model_name:
        return pipeline
    if pipeline is not None:
        pipeline.set_llm_model(model_name)
    else:
        # No pipeline yet (no documents loaded): remember the choice on the shared
        # settings so the pipeline created on first upload/sample-load picks it up.
        SETTINGS.llm_model = model_name
    return pipeline


def handle_connection_check():
    """Re-checked on every page load, not just once at process startup, so the
    banner reflects whether Ollama is reachable right now."""
    connection_error = check_ollama(SETTINGS.ollama_base_url)
    if connection_error:
        return gr.update(value=f"**{connection_error}**", visible=True)
    return gr.update(visible=False)


def handle_message(message: str, history, pipeline: Optional[RAGPipeline], selected_docs: List[str], scope_enabled: bool):
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

    scope = selected_docs if (scope_enabled and selected_docs) else None
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield history, pipeline, ""

    last_docs = []
    try:
        for partial_answer, docs in pipeline.query_stream(message, scope=scope):
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


def handle_regenerate(history, pipeline: Optional[RAGPipeline], selected_docs: List[str], scope_enabled: bool):
    """Re-ask the most recent question, discarding its previous answer.

    Finds the last user turn in the chat history, drops everything after it
    (the stale answer and any excerpts panel), pops the matching turn from
    conversation memory so it is not duplicated, then streams a fresh answer
    exactly as handle_message does.
    """
    history = history or []
    last_user_index = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            last_user_index = i
            break
    if last_user_index is None or pipeline is None or pipeline.index is None:
        yield history, pipeline
        return

    question = _message_text(history[last_user_index]["content"])
    history = history[: last_user_index + 1] + [{"role": "assistant", "content": ""}]
    pipeline.memory.pop_last_turn()
    yield history, pipeline

    scope = selected_docs if (scope_enabled and selected_docs) else None
    last_docs = []
    try:
        for partial_answer, docs in pipeline.query_stream(question, scope=scope):
            last_docs = docs
            history[-1] = {"role": "assistant", "content": partial_answer}
            yield history, pipeline
    except Exception as exc:
        history[-1] = {"role": "assistant", "content": f"Error while generating a response: {exc}"}
        yield history, pipeline
        return

    if last_docs:
        excerpts = "\n\n".join(format_excerpt(doc) for doc in last_docs)
        history = history + [
            {"role": "assistant", "content": excerpts, "metadata": {"title": "View retrieved excerpts"}}
        ]
        yield history, pipeline


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
    # theme/css are applied at launch() time (see main.py), not here: Gradio 6.x
    # moved both parameters from the Blocks constructor to launch().
    with gr.Blocks(title="RAG Study Assistant") as demo:
        gr.HTML(
            "<div id='hero'>"
            "<h1>RAG Study Assistant</h1>"
            "<p>Upload your own PDF, .txt or .docx study documents, or load the bundled History and "
            "Philosophy of Religion sample pack. Every answer is grounded in your material and cites its "
            "source; a question outside that material is declined rather than guessed at.</p>"
            "<span class='badge'>Runs entirely offline &middot; nothing leaves this machine</span>"
            "</div>"
        )
        connection_banner = gr.Markdown(visible=False)
        pipeline_state = gr.State(None)

        # Two-pane app shell: a narrow sidebar for document/session/settings controls,
        # and a wide chat pane, rather than stacking every control in one column.
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320, elem_id="sidebar"):
                gr.Markdown("Study documents", elem_classes=["sidebar-heading"])
                file_upload = gr.Files(label="Upload study documents", file_types=[".pdf", ".txt", ".docx"])
                sample_btn = gr.Button("Load sample documents (History of Religion)")
                with gr.Row():
                    save_session_btn = gr.Button("Save session", size="sm")
                    resume_session_btn = gr.Button("Resume last session", size="sm")
                remove_docs_btn = gr.Button("Remove all documents", size="sm")
                upload_status = gr.Textbox(label="Index status", interactive=False, lines=2)

                with gr.Accordion("Loaded documents", open=False):
                    doc_list = gr.CheckboxGroup(label="Select document(s)", choices=[])
                    scope_checkbox = gr.Checkbox(
                        label="Answer using only the selected document(s) above",
                        value=False,
                        info="Leave unchecked to search all loaded documents.",
                    )
                    remove_selected_btn = gr.Button("Remove selected", size="sm")

                with gr.Accordion("Advanced settings", open=False):
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, step=1, value=SETTINGS.top_k,
                        label="Chunks retrieved per question (top_k)",
                        info="Higher values give the model more context per question, at the cost of a longer prompt. "
                             "Applies to the next question asked.",
                    )
                    model_dropdown = gr.Dropdown(
                        label="Generation model",
                        choices=[SETTINGS.llm_model],
                        value=SETTINGS.llm_model,
                        info="Models currently pulled in the local Ollama server. The embedding model stays fixed, "
                             "so switching this does not require re-embedding any loaded documents.",
                    )
                    answer_style_radio = gr.Radio(
                        choices=[("Concise", "concise"), ("Detailed", "detailed")],
                        value=SETTINGS.answer_style,
                        label="Answer style",
                        info="Detailed answers cover more background and reasoning; concise answers stay short and direct. "
                             "Applies to the next question asked.",
                    )

                with gr.Accordion("Why does it decline some questions?", open=False):
                    gr.Markdown(
                        "This assistant only answers from the documents you have loaded, so every answer can be "
                        "traced back to a source. If you ask something the material does not cover, even a closely "
                        "related follow-up, it will tell you the question is out of scope instead of filling the "
                        "gap with the model's own general knowledge. That guarantee is what makes the citations "
                        "meaningful: an answer is either backed by your material or it is declined, never guessed. "
                        "To widen what it can answer, upload documents that cover the topic you want to ask about."
                    )

                with gr.Accordion("What is this a good fit for?", open=False):
                    gr.Markdown(
                        "Best suited to tasks where every answer needs to trace back to a specific document you "
                        "already trust: revising from your own lecture notes or textbook chapters, checking what a "
                        "particular policy or procedure document actually says, or summarising material you can "
                        "then verify yourself against the citation.\n\n"
                        "It is not a substitute for professional advice in law, medicine, finance or other "
                        "high-stakes domains. Grounding answers in retrieved text and citing sources reduces the "
                        "risk of an unsupported answer, but does not eliminate it: retrieval can still surface the "
                        "wrong passage, and the model can still misread the context it is given. Anything that "
                        "matters should be checked against the cited source, not taken on trust."
                    )

            with gr.Column(scale=2, elem_id="main-chat"):
                chatbot = gr.Chatbot(label="Conversation", height=560, elem_id="chatbot")
                question_box = gr.Textbox(label="Ask a question", placeholder="e.g. What are the Five Pillars of Islam?")
                with gr.Column(visible=True) as examples_col:
                    gr.Examples(examples=SAMPLE_QUESTIONS, inputs=question_box, label="Example questions (after loading the sample documents)")
                with gr.Row():
                    submit_btn = gr.Button("Ask", variant="primary")
                    regenerate_btn = gr.Button("Regenerate answer")
                with gr.Row():
                    clear_btn = gr.Button("Clear conversation")
                    download_btn = gr.DownloadButton("Download transcript", size="sm")

        file_upload.upload(
            handle_upload,
            inputs=[file_upload, pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list, examples_col],
        )
        sample_btn.click(
            handle_load_samples,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list, examples_col],
        )
        remove_docs_btn.click(
            handle_remove_documents,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, chatbot, doc_list, examples_col],
        )
        remove_selected_btn.click(
            handle_remove_selected,
            inputs=[doc_list, pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list, examples_col],
        )
        save_session_btn.click(
            handle_save_session,
            inputs=[pipeline_state],
            outputs=[upload_status],
        )
        resume_session_btn.click(
            handle_resume_session,
            inputs=[pipeline_state],
            outputs=[upload_status, pipeline_state, doc_list, examples_col],
        )
        top_k_slider.change(
            handle_top_k_change,
            inputs=[top_k_slider, pipeline_state],
            outputs=[pipeline_state],
        )
        model_dropdown.change(
            handle_model_change,
            inputs=[model_dropdown, pipeline_state],
            outputs=[pipeline_state],
        )
        answer_style_radio.change(
            handle_answer_style_change,
            inputs=[answer_style_radio, pipeline_state],
            outputs=[pipeline_state],
        )
        submit_btn.click(
            handle_message,
            inputs=[question_box, chatbot, pipeline_state, doc_list, scope_checkbox],
            outputs=[chatbot, pipeline_state, question_box],
        )
        question_box.submit(
            handle_message,
            inputs=[question_box, chatbot, pipeline_state, doc_list, scope_checkbox],
            outputs=[chatbot, pipeline_state, question_box],
        )
        regenerate_btn.click(
            handle_regenerate,
            inputs=[chatbot, pipeline_state, doc_list, scope_checkbox],
            outputs=[chatbot, pipeline_state],
        )
        clear_btn.click(handle_clear, inputs=[pipeline_state], outputs=[chatbot, pipeline_state])
        download_btn.click(handle_download_transcript, inputs=[chatbot], outputs=[download_btn])
        demo.load(handle_connection_check, outputs=[connection_banner])
        demo.load(handle_model_choices, outputs=[model_dropdown])

        gr.HTML("<footer id='footer-note'>Local RAG Study Assistant &middot; Ollama, FAISS and LangChain</footer>")

    return demo


