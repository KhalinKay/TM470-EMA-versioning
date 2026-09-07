import tempfile
from pathlib import Path

from src.rag_assistant.config import SETTINGS, Settings
from src.rag_assistant.pipeline import RAGPipeline
from src.rag_assistant.ui import _examples_visibility, handle_answer_style_change, handle_model_change
from tests.fakes import FakeEmbeddings, FakeLLM


def _pipeline() -> RAGPipeline:
    return RAGPipeline(settings=Settings(), embeddings=FakeEmbeddings(), llm=FakeLLM())


def test_examples_visible_when_no_pipeline_exists_yet():
    assert _examples_visibility(None)["visible"] is True


def test_examples_visible_when_pipeline_has_no_documents_loaded():
    assert _examples_visibility(_pipeline())["visible"] is True


def test_examples_visible_when_exactly_the_shipped_sample_pack_is_loaded():
    pipeline = _pipeline()
    sample_paths = [
        "data/sample_docs/abrahamic_traditions.txt",
        "data/sample_docs/eastern_traditions.txt",
        "data/sample_docs/religion_ancient_origins.txt",
    ]
    pipeline.ingest(sample_paths)

    assert _examples_visibility(pipeline)["visible"] is True


def test_examples_hidden_when_a_custom_document_is_loaded():
    """The shipped example questions name sample-pack specifics (Akhenaten, the
    Council of Nicaea, ...), so they should not be shown once a tutor's own,
    unrelated material is loaded instead."""
    pipeline = _pipeline()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "organic_chemistry.txt"
        path.write_text("Organic chemistry studies carbon-containing compounds. " * 10, encoding="utf-8")
        pipeline.ingest([str(path)])

    assert _examples_visibility(pipeline)["visible"] is False


def test_handle_model_change_remembers_choice_on_settings_when_no_pipeline_yet():
    """A model picked before any document is loaded (pipeline is still None) must
    still take effect once a pipeline is created, via the shared settings object."""
    original_model = SETTINGS.llm_model
    try:
        result = handle_model_change("dolphin3:latest", None)
        assert result is None
        assert SETTINGS.llm_model == "dolphin3:latest"
    finally:
        SETTINGS.llm_model = original_model


def test_handle_model_change_updates_an_existing_pipeline():
    settings = Settings()
    pipeline = RAGPipeline(settings=settings, embeddings=FakeEmbeddings(), llm=FakeLLM())
    original_llm = pipeline.llm

    result = handle_model_change("dolphin3:latest", pipeline)

    assert result is pipeline
    assert pipeline.settings.llm_model == "dolphin3:latest"
    assert pipeline.llm is not original_llm


def test_handle_answer_style_change_remembers_choice_on_settings_when_no_pipeline_yet():
    original_style = SETTINGS.answer_style
    try:
        result = handle_answer_style_change("detailed", None)
        assert result is None
        assert SETTINGS.answer_style == "detailed"
    finally:
        SETTINGS.answer_style = original_style


def test_handle_answer_style_change_updates_an_existing_pipeline():
    settings = Settings()
    pipeline = RAGPipeline(settings=settings, embeddings=FakeEmbeddings(), llm=FakeLLM())

    result = handle_answer_style_change("detailed", pipeline)

    assert result is pipeline
    assert pipeline.settings.answer_style == "detailed"
