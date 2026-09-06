from src.rag_assistant.config import SETTINGS, Settings
from src.rag_assistant.pipeline import RAGPipeline
from src.rag_assistant.ui import handle_answer_style_change, handle_model_change
from tests.fakes import FakeEmbeddings, FakeLLM


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
