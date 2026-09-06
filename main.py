"""Launch entrypoint for the RAG Study Assistant Gradio app."""
from src.rag_assistant.ui import build_app, THEME


def main() -> None:
    app = build_app()
    app.launch(theme=THEME)


if __name__ == "__main__":
    main()
