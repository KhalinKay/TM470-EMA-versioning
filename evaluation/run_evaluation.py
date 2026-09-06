"""TruLens RAG Triad evaluation across chunk sizes 256, 512 and 1024 characters.

Compares Groundedness, Answer Relevance and Context Relevance (Lewis et al.,
2020; Gao et al., 2023) using a locally hosted Ollama model as the LLM judge
via TruLens's LiteLLM provider, keeping the whole evaluation offline.

Usage:
    python -m evaluation.run_evaluation --sample-docs data/sample_docs
"""
import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trulens.core import TruSession, Metric, Selector
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp
from trulens.providers.litellm import LiteLLM
from langchain_ollama import ChatOllama

from src.rag_assistant.config import SETTINGS
from src.rag_assistant.ingestion import load_documents, split_documents
from src.rag_assistant.indexing import get_embeddings, build_index
from src.rag_assistant.prompts import SYSTEM_PROMPT, CONCISE_INSTRUCTION
from evaluation.test_questions import TEST_QUESTIONS

CHUNK_SIZES = [256, 512, 1024]
CHUNK_OVERLAP_RATIO = 0.125  # matches the evaluated 64/512 ratio, scaled per chunk size
METRIC_NAMES = ["Groundedness", "Answer Relevance", "Context Relevance"]


class EvaluatedRAG:
    """Thin retrieve/generate wrapper, instrumented so TruLens can trace and score it."""

    def __init__(self, index, llm):
        self.index = index
        self.llm = llm

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> List[str]:
        docs = self.index.similarity_search(query, k=SETTINGS.top_k)
        return [doc.page_content for doc in docs]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, context_list: List[str]) -> str:
        context = "\n\n".join(context_list)
        prompt = SYSTEM_PROMPT.format(
            context=context,
            history="(no previous conversation)",
            question=query,
            length_instruction=CONCISE_INSTRUCTION,
        )
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", response)).strip()

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        context_list = self.retrieve(query)
        return self.generate(query, context_list)


def build_feedback_functions(provider: LiteLLM):
    f_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
        name="Groundedness",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
            "question": Selector.select_record_input(),
        },
    )
    f_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )
    f_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="Context Relevance",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )
    return [f_groundedness, f_answer_relevance, f_context_relevance]


def _sample_doc_paths(sample_docs_dir: Path) -> List[str]:
    paths = [str(p) for p in sample_docs_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"}]
    if not paths:
        raise FileNotFoundError(
            f"No sample documents found in {sample_docs_dir}. Add PDF, .txt or .docx files there first."
        )
    return paths


def run_for_chunk_size(chunk_size: int, sample_docs_dir: Path, session: TruSession, provider: LiteLLM) -> dict:
    overlap = max(8, int(chunk_size * CHUNK_OVERLAP_RATIO))
    documents = load_documents(_sample_doc_paths(sample_docs_dir))
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)

    embeddings = get_embeddings(SETTINGS.embedding_model, SETTINGS.ollama_base_url)
    index = build_index(chunks, embeddings)
    llm = ChatOllama(model=SETTINGS.llm_model, base_url=SETTINGS.ollama_base_url, temperature=0.0)

    rag = EvaluatedRAG(index, llm)
    feedbacks = build_feedback_functions(provider)
    tru_rag = TruApp(
        rag,
        app_name="RAGStudyAssistant",
        app_version=f"chunk_{chunk_size}",
        feedbacks=feedbacks,
    )

    with tru_rag as recording:
        for item in TEST_QUESTIONS:
            rag.query(item["question"])

    leaderboard = session.get_leaderboard(app_ids=[tru_rag.app_id])
    return {
        metric: float(leaderboard[metric].mean()) if metric in leaderboard.columns else float("nan")
        for metric in METRIC_NAMES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG chunk sizes using the TruLens RAG Triad.")
    parser.add_argument("--sample-docs", default="data/sample_docs", help="Directory of documents to evaluate against.")
    parser.add_argument("--output-dir", default="data/eval_results", help="Directory for the results table and chart.")
    parser.add_argument("--judge-model", default="llama3:8b", help="Ollama model used as the local TruLens judge.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = TruSession()
    session.reset_database()
    provider = LiteLLM(model_engine=f"ollama/{args.judge_model}", api_base=SETTINGS.ollama_base_url)

    rows = []
    for chunk_size in CHUNK_SIZES:
        print(f"Evaluating chunk size {chunk_size}...")
        means = run_for_chunk_size(chunk_size, Path(args.sample_docs), session, provider)
        rows.append({"chunk_size": chunk_size, **means})

    results = pd.DataFrame(rows).set_index("chunk_size")
    print("\nResults:\n", results)
    results.to_csv(output_dir / "chunk_size_comparison.csv")

    ax = results.plot(kind="bar", ylim=(0, 1), title="RAG Triad scores by chunk size")
    ax.set_ylabel("Score (0-1)")
    plt.tight_layout()
    plt.savefig(output_dir / "chunk_size_comparison.png")

    print(f"\nSaved table to {output_dir / 'chunk_size_comparison.csv'}")
    print(f"Saved chart to {output_dir / 'chunk_size_comparison.png'}")


if __name__ == "__main__":
    main()
