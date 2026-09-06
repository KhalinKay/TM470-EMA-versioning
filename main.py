"""Launch entrypoint for the RAG Study Assistant Gradio app."""
from src.rag_assistant.ui import build_app, THEME, CUSTOM_CSS


def main() -> None:
    app = build_app()
    app.launch(theme=THEME, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
