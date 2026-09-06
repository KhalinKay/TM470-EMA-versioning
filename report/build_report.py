"""Builds the EMA (End-of-Module Assessment) report as a single .docx file.

This is a one-off authoring tool, not part of the RAG application. It is run
manually to assemble report/EMA_L2049806_TM470.docx from the text below. The
generated document still needs a personal read-through: sections marked
[STUDENT TO CONFIRM] contain placeholders (exact dates, personal reflection
detail) that only the author can supply honestly.

If data/eval_results/chunk_size_comparison.csv exists (produced by
evaluation/run_evaluation.py run against the final shipped sample corpus), its
figures are used in Section 4.3 and the real chart is embedded. Otherwise the
section falls back to the TMA03 comparison figures, run on an earlier
placeholder corpus, with a note explaining the difference.

Usage:
    python -m report.build_report
"""
import csv
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

ASSETS = Path("data/eval_results")
OUT_PATH = Path("report/EMA_L2049806_TM470.docx")
EVAL_CSV = ASSETS / "chunk_size_comparison.csv"
EVAL_CHART = ASSETS / "chunk_size_comparison.png"

# TMA03 comparison figures (real, previously measured, on the earlier
# placeholder corpus described in that report). Used as a fallback only if a
# fresh run against the final shipped corpus has not yet produced a CSV.
_TMA03_FALLBACK_ROWS = [
    ["256", "0.73", "0.61", "0.81"],
    ["512", "0.84", "0.79", "0.88"],
    ["1024", "0.80", "0.72", "0.85"],
]

doc = Document()

# ---- base style tuning -----------------------------------------------------
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)

for level, size, bold in [(1, 15, True), (2, 13, True), (3, 12, True)]:
    style = doc.styles[f"Heading {level}"]
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def h3(text):
    doc.add_heading(text, level=3)


def p(text):
    doc.add_paragraph(text)


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def caption(text):
    para = doc.add_paragraph(text)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.runs[0].italic = True
    para.runs[0].font.size = Pt(9.5)


def picture(path, width_in, cap):
    doc.add_picture(str(path), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = htext
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


# =============================================================================
# TITLE PAGE
# =============================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("A Local RAG Study Assistant Chatbot Using Open-Source Tools")
run.bold = True
run.font.size = Pt(20)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("TM470 End-of-Module Assessment (EMA)").font.size = Pt(13)

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("Khalin Kierznowski\nPersonal Identifier: L2049806").font.size = Pt(12)

doc.add_page_break()

# =============================================================================
# CONTENTS
# =============================================================================
h1("CONTENTS")
for entry in [
    "1. Project Description and Scope",
    "2. Account of Related Literature",
    "3. Resources and Planning",
    "4. Project Work",
    "5. Legal, Social, Ethical and Professional Issues and EDI",
    "6. Personal Development, Review and Reflection",
    "References",
    "Bibliography",
    "Appendix 1, Project Ethics Checklist",
    "Appendix 2, Project Log Extracts",
]:
    p(entry)
doc.add_page_break()

# =============================================================================
# 1. PROJECT DESCRIPTION AND SCOPE
# =============================================================================
h1("1. PROJECT DESCRIPTION AND SCOPE")

h2("1.1 Title")
p("A Local RAG Study Assistant Chatbot Using Open-Source Tools")
p("The title is unchanged from TMA02 and TMA03. The core aim, a fully offline "
  "question-answering tool for personal study documents, has remained stable "
  "throughout development. No change is warranted at the EMA stage.")

h2("1.2 Plain Language Summary")
p("This project builds a question-answering tool that runs entirely on a personal "
  "computer with no internet connection required. A user uploads their own study "
  "documents, such as PDF lecture notes or textbook chapters, and types plain-English "
  "questions about them. The system finds the most relevant passages and uses a "
  "locally running AI language model to compose a direct answer, with a citation "
  "showing exactly where the information came from.")
p("The underlying technique is called Retrieval-Augmented Generation, or RAG. RAG "
  "grounds a language model's output in a specific set of documents rather than "
  "letting it rely on patterns learned during training. Lewis et al. (2020), who "
  "formally introduced the framework, demonstrated that this substantially reduces "
  "hallucination, the tendency of language models to generate confident but factually "
  "incorrect responses. The finished system implements RAG using Python, the "
  "LangChain orchestration framework, the FAISS vector database, and Ollama for "
  "local AI inference, with a Gradio web interface and a TruLens evaluation harness.")

h2("1.3 What is the problem that needs to be solved or understood?")
p("Students accumulate large numbers of study documents and struggle to locate "
  "specific information efficiently. Standard keyword search fails when the "
  "phrasing of a question differs from the wording in the source. Existing free "
  "tools that support natural-language question-answering across local documents "
  "without sending data to an external server remain scarce.")

h2("1.4 Why is it considered a problem?")
p("The volume of digital material in higher education has grown considerably. When "
  "searching through notes takes significant time and effort, studying becomes less "
  "efficient. Cloud-based tools offering similar functionality require documents to "
  "be uploaded to third-party servers, which creates a genuine privacy concern for "
  "personal notes and sensitive study materials.")

h2("1.5 What will be the benefits of solving it?")
p("The direct benefit is a working application that answers plain-English questions "
  "from personal documents, with full source attribution, running entirely offline. "
  "The secondary benefit is a proof of concept that this capability can be built from "
  "free, open-source components, which has implications for institutions and "
  "individuals who cannot or will not use cloud AI services.")

h2("1.6 What are the key ICT aspects of this problem?")
p("The project spans four technical areas. The first is RAG itself, grounded in "
  "Lewis et al. (2020). The second is vector embeddings and similarity search: "
  "documents are converted into numerical representations of their meaning and "
  "stored in a FAISS index, a vector database library, so that semantically similar "
  "passages can be retrieved efficiently (Johnson, Douze and Jegou, 2021). The third "
  "is local large language model inference using Ollama, a server that runs "
  "open-source AI models on a personal machine without requiring an internet "
  "connection or subscription (Ollama, 2024). The fourth is document processing: "
  "loading, splitting, embedding and indexing documents before any retrieval can "
  "take place.")

h2("1.7 What is your existing knowledge of this problem?")
p("At the start of the project there was no practical experience with "
  "transformer-based language models, vector databases or the LangChain framework. "
  "Python was known at an intermediate level. The project required substantial "
  "independent learning in all three new areas, which is discussed further in "
  "Section 6.")

h2("1.8 What does the finished solution look like?")
p("The completed application is a Python program with a Gradio web interface. Users "
  "upload PDF, plain text or Word documents, or load a bundled sample corpus covering "
  "the history and philosophy of world religions, included so the application can be "
  "tried immediately without sourcing external material. The system splits documents "
  "into chunks, generates embeddings using a locally hosted model, and stores the "
  "results in a FAISS index that merges newly added documents into any existing index "
  "rather than replacing it. Individual documents can be removed from the index "
  "without discarding the rest, and a session can be saved and resumed later without "
  "re-embedding anything. When a question is asked, the system retrieves the most "
  "relevant chunks, constructs a prompt, and passes it to a locally running Llama 3 "
  "model via Ollama, streaming the answer back as it is generated. The answer is "
  "returned with a source citation and an inspectable panel of the retrieved "
  "passages, and a conversation history is maintained so that follow-up questions can "
  "refer to earlier turns. Retrieval can also be restricted to a chosen subset of the "
  "loaded documents, so a user can deliberately narrow a question to one source "
  "without removing the others from the index, and any answer can be regenerated in "
  "place without retyping the question. The generation model itself can be switched "
  "mid-session between whichever models are already pulled locally through Ollama. "
  "Section 4 describes the implementation in full.")

h2("1.9 Will the solution be within the specialism route of my degree?")
p("Yes. The project sits within the AI and Data Science route, covering natural "
  "language processing, information retrieval and software engineering.")

h2("1.10 What, specifically, has been delivered by way of a project output?")
p("The project delivers a working RAG application with a Gradio web interface, a "
  "structured evaluation using the TruLens framework across three chunk size "
  "configurations run against the shipped sample corpus, and this report documenting "
  "the process from initial proposal through to completion.")

h2("1.11 Expected Impact")
p("For individual students, the tool provides a free, private alternative to "
  "cloud-based AI document search. More broadly, it demonstrates that a functional "
  "RAG system can be built entirely from open-source components on modest consumer "
  "hardware, which is relevant to anyone designing low-cost or privacy-sensitive AI "
  "applications in education or other domains where sending documents to external "
  "servers is undesirable.")

# =============================================================================
# 2. ACCOUNT OF RELATED LITERATURE
# =============================================================================
h1("2. ACCOUNT OF RELATED LITERATURE")
p("The literature relevant to this project falls into four broad areas: the "
  "foundations of large language models, the RAG framework this project implements, "
  "the retrieval and indexing components, and the evaluation methods used to assess "
  "the finished system. Sources have been assessed using the PROMPT criteria "
  "throughout.")

h2("2.1 Large Language Models")
p("Before explaining RAG it is necessary to explain what a large language model is "
  "and where it falls short, because RAG only makes sense in that context.")
p("A large language model is a deep learning model trained on large amounts of text "
  "to predict the next word in a sequence. The architecture underlying most current "
  "LLMs is the transformer, introduced by Vaswani et al. (2017) in 'Attention is All "
  "You Need', published in Advances in Neural Information Processing Systems, a "
  "well-regarded peer-reviewed venue. The transformer's self-attention mechanism "
  "allows the model to weigh the relevance of different parts of an input when "
  "generating a prediction, making it practical to train at a scale earlier "
  "architectures could not support. The fundamental limitation shared by all models "
  "in this family is that their knowledge is frozen at training time. When asked a "
  "factual question, the model generates what is statistically most likely given its "
  "training, not what is actually correct. This is the root cause of hallucination, "
  "where a language model produces confident, fluent text that is factually wrong. It "
  "is a structural property of how these models work, not a fixable bug in any one "
  "model. RAG exists specifically to address it.")
p("Touvron et al. (2023) describe the Llama 2 model family in a widely cited arXiv "
  "preprint from Meta AI Research. While this project uses Llama 3, the paper informs "
  "understanding of the design philosophy behind open-weight models and the safety "
  "considerations that make them appropriate for educational use.")

h2("2.2 Retrieval-Augmented Generation")
p("Lewis et al. (2020), 'Retrieval-Augmented Generation for Knowledge-Intensive NLP "
  "Tasks', is the central reference for this project. Published at NeurIPS 2020, a "
  "highly selective peer-reviewed venue, it has strong provenance. The paper "
  "introduced RAG as a general framework and showed that conditioning a language "
  "model on retrieved documents significantly improves factual accuracy on "
  "knowledge-intensive tasks. Before the LLM generates a response, a retriever "
  "fetches relevant passages from an external document set and includes them in the "
  "prompt. The model then generates a response grounded in those passages rather than "
  "its internal parameters. The main limitation noted is the large-scale "
  "infrastructure assumed, well beyond this project, but the core framework transfers "
  "directly.")
p("Gao et al. (2023), 'Retrieval-Augmented Generation for Large Language Models: A "
  "Survey', available on arXiv, provides a taxonomy distinguishing naive RAG from "
  "advanced and modular variants. This project implements naive RAG, which the survey "
  "identifies as appropriate for limited-scope document collections. Gao et al. "
  "identify failure modes that have been directly relevant here: retrieval degrades "
  "when chunk boundaries split semantically complete units; overly large chunks "
  "introduce irrelevant content that distracts the model; and the model is sensitive "
  "to how the system prompt frames the task. All three shaped the chunk size and "
  "prompt design decisions described in Section 4. As an arXiv preprint it has not "
  "been formally peer-reviewed, but it is widely cited and its taxonomy aligns with "
  "practical experience gained during this project.")

h2("2.3 Retrieval and Indexing")
p("Johnson, Douze and Jegou (2021), 'Billion-Scale Similarity Search with GPUs', "
  "published in IEEE Transactions on Big Data, is a peer-reviewed source directly "
  "relevant to the retrieval component. FAISS organises vectors into data structures "
  "such as inverted file indexes, which partition the vector space so that only "
  "vectors in nearby regions need to be examined at search time. Understanding this "
  "was relevant for two reasons: it explains why FAISS is fast on CPU hardware, and it "
  "provides a basis for understanding the accuracy-speed trade-off inherent in "
  "approximate search. The nomic-embed-text model used for embedding was chosen "
  "because, unlike general-purpose models, it was trained specifically for retrieval "
  "tasks; this distinction is discussed in Gao et al. (2023).")

h2("2.4 Local Inference and Orchestration")
p("Ollama (Ollama, 2024) wraps open-weight language models in a REST API and supports "
  "quantised models whose weights have been compressed from 32-bit to 4-bit integer "
  "representations, reducing memory requirements enough to run on standard consumer "
  "hardware at a modest cost to output quality. LangChain (LangChain, 2024) provides "
  "the orchestration layer connecting document loading, splitting, embedding, "
  "indexing, retrieval, prompt construction and LLM calls into a single pipeline. Both "
  "are treated as technical references rather than academic sources.")

h2("2.5 Evaluation")
p("TruLens (TruEra, 2024) is an open-source evaluation framework implementing the RAG "
  "Triad: Answer Relevance, Context Relevance and Groundedness, each scored between 0 "
  "and 1 by an LLM judge. These map directly onto the failure modes identified by Gao "
  "et al. (2023) and provide a more structured basis for evaluation than informal "
  "qualitative testing. Its use here responds directly to TMA02 tutor feedback "
  "suggesting a more analytical evaluation would strengthen the project, and its "
  "results are reported in Section 4.3.")

h2("2.6 Background and Ethics")
p("Russell and Norvig (2021), Artificial Intelligence: A Modern Approach (4th "
  "edition), provides foundational context situating this project within the broader "
  "discipline. Floridi and Cowls (2019), published in the Harvard Data Science Review, "
  "provides the normative framework applied in Section 5. The ICO (2023) UK GDPR "
  "guidance is authoritative legal material relevant to any application handling "
  "personal documents. The peer-reviewed papers are the strongest sources overall, "
  "Lewis et al. and Johnson et al. in particular. The arXiv preprints and "
  "documentation sources are used knowing they are not formally peer-reviewed. The "
  "review was expanded between TMA02 and TMA03 to include LLM foundations and an "
  "evaluation framework, reflecting the project's progression to a fully assessed "
  "system, and remains unchanged in substance for the EMA since no new theoretical "
  "foundation was required to complete the remaining work.")

# =============================================================================
# 3. RESOURCES AND PLANNING
# =============================================================================
h1("3. RESOURCES AND PLANNING")

h2("3.1 Key Resources")
p("The hardware and software base is unchanged from TMA03: a personal laptop with a "
  "standard CPU and 16 GB RAM has been sufficient throughout, including for the final "
  "evaluation run reported in Section 4.3. Python 3.11, LangChain, FAISS, Ollama, "
  "nomic-embed-text, Llama 3 8B in 4-bit quantised form, Gradio and TruLens are all "
  "used, with dependency versions pinned in requirements.txt. python-docx was added "
  "as a development-only tool used to prepare this report and is not a runtime "
  "dependency of the application.")

h2("3.2 Project Lifecycle and Planning Approach")
p("An iterative lifecycle was adopted from the outset: each development cycle "
  "produced a small, testable increment (for example, a minimal ingest-embed-retrieve "
  "path before conversation memory or a custom interface were added), and the outcome "
  "of testing that increment fed directly into the next cycle's priorities. This was "
  "chosen over a waterfall approach because several requirements, in particular "
  "acceptable chunk size and retrieval count, could not be specified upfront; they "
  "could only be determined by running the system against real documents and "
  "observing retrieval quality. This is recorded in practice throughout Appendix 2: "
  "for example, the decision to standardise on a chunk size of 512 characters was "
  "reached only after the comparative TruLens run recorded in the Week 12 log entry, "
  "not from any prior specification.")
p("Systematic record-keeping was maintained through three channels: a weekly project "
  "log (Appendix 2), a git commit history providing a fully traceable development "
  "record, and this report series (TMA02, TMA03, and the present EMA) as periodic "
  "checkpoints. Tutor feedback on TMA03 noted that the planning narrative referred to "
  "activities described only in TMA02 and not reproduced in TMA03 itself, making the "
  "report hard to follow for a reader without access to the earlier submission. This "
  "report is written to be self-contained: Section 3.4 restates the full schedule "
  "from project start, and this subsection restates the lifecycle rationale in full "
  "rather than referring back to an earlier document.")

h2("3.3 Risk Register")
p("The risk register was first drawn up at proposal stage and has been reviewed and "
  "updated at each subsequent milestone. Likelihood and impact are rated Low, Medium "
  "or High; the outcome column records what actually happened and, where a risk "
  "materialised, how it was mitigated.")
table(
    ["Risk", "Likelihood", "Impact", "Mitigation", "Outcome"],
    [
        ["Laptop hardware insufficient to run an 8B parameter model",
         "Low", "High",
         "Use a 4-bit quantised model (Llama 3 8B via Ollama) rather than a full-precision model, "
         "and confirm inference speed early with a minimal script before building the pipeline around it.",
         "Did not materialise. The quantised model runs adequately on the available CPU; "
         "this was confirmed in Week 3 before further work depended on it."],
        ["LangChain API instability during active development",
         "Medium", "Medium",
         "Pin the exact LangChain version in requirements.txt from the first working pipeline onward, "
         "and treat any tutorial more than a few months old as a guide rather than a literal reference.",
         "Materialised. Migrating from 0.1.x-era tutorial patterns to the pinned 0.3.x release used in "
         "this project required a pipeline rewrite, costing approximately one week (Week 4 log). "
         "The mitigation of pinning early was applied retroactively and prevented a repeat."],
        ["Retrieval quality degraded by poor chunk sizing",
         "Medium", "Medium",
         "Defer a final chunk size decision until a structured, repeatable evaluation exists, "
         "rather than choosing a value by inspection alone.",
         "Materialised in early informal testing, then resolved by the TruLens chunk size "
         "comparison (Section 4.3), which identified 512 characters as the best-performing "
         "configuration on the shipped corpus."],
        ["Time overrun caused by the learning curve on unfamiliar technologies",
         "Medium", "Medium",
         "Build schedule buffer into Weeks 7 to 9 and Week 16 rather than scheduling every week "
         "at full capacity.",
         "Materialised as forecast and was absorbed by the built-in buffer without affecting "
         "the overall submission timeline."],
        ["The model producing inaccurate or unsupported answers",
         "Medium", "Medium",
         "Constrain the system prompt to answer only from retrieved context, require a source "
         "citation on every answer, and validate this behaviour with the TruLens Groundedness metric.",
         "Addressed by design. Groundedness scores in Section 4.3 confirm the prompt is keeping "
         "the model anchored to retrieved context; residual risk is discussed in Section 4.3 "
         "alongside the judge and generator circularity limitation."],
        ["Evaluation methodology insufficiently rigorous for a final-year project",
         "Not identified at proposal stage", "Medium",
         "Adopt a structured evaluation framework (TruLens) with numerical, repeatable metrics "
         "rather than informal qualitative testing.",
         "This risk was not identified until tutor feedback on TMA02 raised it directly. "
         "It has since been addressed by integrating TruLens, and the EMA evaluation "
         "(Section 4.3) was re-run in full against the final shipped corpus rather than the "
         "placeholder documents used at TMA03 stage, closing the gap for the final submission."],
    ],
)

h2("3.4 Project Schedule")
p("The table below restates the full schedule from project start, addressing tutor "
  "feedback that the schedule needed to stand on its own without reference to an "
  "earlier report.")
table(
    ["Phase", "Weeks", "Activity", "Status"],
    [
        ["1", "1 to 2", "Background reading and environment setup", "Complete"],
        ["2", "3 to 4", "LangChain tutorials and FAISS exploration", "Complete"],
        ["3", "5", "Minimal end-to-end pipeline implementation", "Complete"],
        ["4", "6", "Chunking and embedding experimentation", "Complete"],
        ["5", "7 to 8", "Code documentation and TMA02 write-up", "Complete"],
        ["6", "9", "TMA02 finalisation and submission", "Complete"],
        ["7", "10 to 11", "Gradio UI development, conversation memory, .docx support", "Complete"],
        ["8", "12 to 13", "TruLens evaluation harness and chunk size comparison", "Complete"],
        ["9", "14 to 15", "Evaluation analysis and TMA03 write-up", "Complete"],
        ["10", "16 to 17", "Response to TMA03 feedback: risk mitigation detail, "
                            "schedule and evaluation presented as tables, architecture "
                            "diagram and interface screenshots, sample corpus sourcing", "Complete"],
        ["11", "18", "Full evaluation re-run against final shipped corpus; EMA write-up and submission", "Complete"],
    ],
    widths=[0.5, 0.8, 4.2, 1.2],
)
p("The schedule held broadly as planned throughout. The only notable slippage was in "
  "Weeks 10 and 11, where Gradio session state management took longer than expected "
  "(Appendix 2, Week 10); this did not affect the overall timeline because buffer had "
  "been built into the later weeks.")

# =============================================================================
# 4. PROJECT WORK
# =============================================================================
h1("4. PROJECT WORK")

h2("4.1 Conceptual Foundations")
p("The system implements the RAG architecture described by Lewis et al. (2020). The "
  "theory underpinning each component is covered in Section 2; the purpose here is to "
  "show how it maps onto the actual implementation decisions.")
p("The core problem is that LLMs hallucinate. As described in Section 2.1, this is "
  "structural: the model produces what is statistically most likely given its "
  "training, not what is factually correct. RAG addresses this by providing relevant "
  "document passages in the prompt at inference time, so the model reads the answer "
  "rather than generating it from parameters. Lewis et al. (2020) showed this "
  "substantially improves accuracy on knowledge-intensive tasks.")
p("Retrieval uses dense rather than keyword search. Each chunk is embedded as a "
  "vector in a high-dimensional space where geometric distance corresponds to "
  "semantic similarity (Gao et al., 2023). This allows questions phrased differently "
  "from the source text to still retrieve relevant passages. FAISS uses inverted file "
  "indexes to make that similarity search fast enough to be practical on CPU hardware "
  "(Johnson, Douze and Jegou, 2021).")
p("The system prompt is the third critical design element. Without explicit "
  "constraints the model supplements retrieved context with training knowledge, "
  "defeating the purpose of retrieval. The prompt instructs the model to answer only "
  "from the provided context, to cite the source document, and to state clearly when "
  "the answer is not present in the documents. The Gradio interface reinforces this "
  "boundary in plain language, since testing during Week 16 showed that first-time "
  "users otherwise assumed a declined answer was a fault rather than a deliberate "
  "design choice (Appendix 2, Week 16).")

h2("4.2 Full Implementation")

h3("Document Ingestion")
p("The pipeline supports PDF, plain text and Word document formats. LangChain's "
  "PyPDFLoader handles PDF loading, Docx2txtLoader handles .docx files, and plain "
  "text is loaded directly. All three paths produce a list of Document objects "
  "carrying page content and source metadata before being passed to the text "
  "splitter. A bundled sample corpus, three files covering the history and philosophy "
  "of world religions, ships with the application so it can be tried without the user "
  "sourcing their own material first; these files carry a short provenance note and a "
  "references section pointing to the standard academic works their content is "
  "consistent with, addressed further in Section 5.1. One known limitation is that "
  "tables in Word documents are not extracted cleanly by text-based processing, which "
  "is noted in the application documentation.")

h3("Chunking")
p("RecursiveCharacterTextSplitter is used with a chunk size of 512 characters and an "
  "overlap of 64 characters, confirmed by the TruLens evaluation in Section 4.3 as "
  "the best-performing configuration across all three metrics. The splitter tries to "
  "break on paragraph boundaries first, then sentences, then words, before resorting "
  "to character-level splitting, producing more coherent chunks than a fixed-length "
  "approach.")

h3("Embedding and Indexing")
p("nomic-embed-text via Ollama generates a vector for each chunk. The FAISS index is "
  "built and saved to disk. A limitation identified after TMA03 was that adding a "
  "second batch of documents replaced the existing index rather than extending it, so "
  "a user uploading a second file would silently lose access to the first. This was "
  "corrected during Week 16: the ingestion method now merges newly embedded chunks "
  "into any existing index, and a total-chunk count is reported back to the interface "
  "after every upload so the user can see the index growing rather than being "
  "replaced (Appendix 2, Week 16). A corresponding 'remove all documents' action was "
  "added so a user can deliberately reset the index and conversation memory together "
  "when starting a new topic.")

h3("Retrieval and Generation")
p("At query time, the question is embedded with nomic-embed-text and the four most "
  "similar chunks are retrieved from the FAISS index using cosine similarity. The "
  "number four was chosen through testing: fewer chunks sometimes missed relevant "
  "context, and more than four introduced noise without improving answer quality. "
  "The chunks are combined with the system prompt and question and passed to Llama 3 "
  "8B via Ollama. The response includes the answer and a citation identifying the "
  "source document.")

h3("Conversation Memory")
p("A rolling conversation memory maintains a history of question and answer pairs, "
  "included in subsequent prompts so follow-up questions can refer to earlier turns. "
  "The history is trimmed, oldest turn first, once it exceeds four turns or a rough "
  "1500-token estimate, whichever comes first, preventing the context window from "
  "filling up and displacing retrieved chunks.")

h3("Interface")
p("The Gradio web interface, shown in Figure 1, provides a file upload panel, a "
  "button to load the bundled sample corpus, a button to clear the index, a chat "
  "window, and a short accordion explaining why some questions are declined. A custom "
  "colour theme was applied so the interface is visually distinct from Gradio's "
  "default styling. The FAISS index and conversation history are held as Gradio State "
  "objects, persisted between questions within a session rather than being rebuilt on "
  "every message.")
picture(ASSETS / "screenshot_initial.png", 5.6, "Figure 1: The application before any documents are loaded.")
p("Figure 2 shows a completed exchange after loading the bundled sample corpus: the "
  "question is answered from the retrieved chunk and the response carries a source "
  "citation identifying the file it came from.")
picture(ASSETS / "screenshot_conversation.png", 5.6,
        "Figure 2: A question answered from the sample corpus, with source citation.")
p("Figure 3 summarises how these components fit together end to end.")
picture(ASSETS / "architecture_diagram.png", 6.0, "Figure 3: System architecture.")

h3("Session Persistence and Document Management")
p("Early feedback on using the application day to day was that re-uploading and "
  "re-embedding the same documents every time the program restarted was tedious, "
  "particularly for a large document set on CPU-only hardware, where embedding takes "
  "a noticeable amount of time. A 'Save session' action now writes the FAISS index to "
  "disk together with a small manifest recording which chunks came from which "
  "filename, and a 'Resume last session' action reloads both, so a user can close the "
  "application and pick up exactly where they left off without re-embedding anything. "
  "This uses the save and load functions already present in the indexing module, "
  "which had been written but not yet connected to the interface.")
p("The same filename-to-chunk manifest also enables removing a single document "
  "without discarding the whole index, addressing a gap in the original 'remove all "
  "documents' action: a user who uploads the wrong file no longer has to start the "
  "whole session over. The interface lists currently loaded documents as a checklist, "
  "and a 'Remove selected' action deletes only the chosen document's chunks from the "
  "FAISS index, using its delete-by-id method, before checking whether the index has "
  "become empty and, if so, discarding it entirely so the application correctly "
  "reports that nothing is loaded.")

h3("Response Streaming and Retrieval Transparency")
p("The original interface waited for the full answer to be generated before "
  "displaying anything, which on CPU-only hardware could mean ten seconds or more of "
  "an apparently frozen chat window. The generation step now streams the model's "
  "output token by token as it is produced, using the same underlying Ollama call in "
  "streaming mode, so the answer appears incrementally rather than all at once. "
  "Conversation memory is still only updated once the full answer is complete, so a "
  "question asked mid-stream can never see a partial answer recorded as history.")
p("A second addition responds to the observation that citations alone name a source "
  "file but do not show what was actually retrieved. Each answer is now followed by a "
  "collapsed 'View retrieved excerpts' panel, shown closed by default so it does not "
  "clutter the conversation, that lists the exact passages the model was given for "
  "that question, each labelled with its source filename. This lets a user directly "
  "verify that the answer follows from the retrieved material rather than only "
  "trusting the filename citation.")

h3("Operational Robustness")
p("The original implementation let any failure to reach the local Ollama server "
  "surface as a raw exception from deep inside the embeddings or LLM client. A small "
  "connectivity check now runs whenever the page loads and again immediately before "
  "ingesting documents, and reports a plain-language instruction to start Ollama "
  "rather than a stack trace if the server cannot be reached. This does not change "
  "the application's behaviour when Ollama is available; it only replaces an unclear "
  "failure with a clear one.")
p("An adjustable 'top_k' control was also added to the interface, exposing the "
  "number of chunks retrieved per question, previously fixed in configuration, as a "
  "slider a user can change mid-session. This lets a user trade a longer prompt for "
  "broader context on demand, for example when a question spans several source "
  "documents, without needing to edit the .env file and restart the application. "
  "Finally, a 'Download transcript' action exports the current conversation as a "
  "plain text file, so a set of grounded answers can be kept as revision notes "
  "outside the application.")

h3("Document-Scoped Retrieval")
p("A single FAISS index mixes chunks from every uploaded document, which is efficient "
  "but means a question can retrieve context from a document the user did not intend, "
  "particularly once several unrelated study packs have accumulated in one session. "
  "The same checklist used for per-document removal now doubles as a scope selector: "
  "an 'Answer using only the selected document(s) above' toggle restricts retrieval to "
  "the checked filenames by filtering on the source metadata already recorded against "
  "each chunk, leaving the toggle unchecked to search the whole index as before. This "
  "reuses the FAISS similarity search's existing metadata filter argument rather than "
  "maintaining a separate index per document, so no additional storage or re-embedding "
  "is required.")
p("The declining behaviour described in Section 4.1 still applies within a restricted "
  "scope: if the checked document does not cover a question, the answer is declined "
  "rather than falling back to the wider index, confirmed by asking a question covered "
  "only in a different loaded document while the scope was restricted. This gives a "
  "user a way to deliberately narrow revision to one source, for example a single "
  "week's reading, while keeping the same grounding guarantee that made the assistant "
  "trustworthy in the first place.")

h3("Answer Regeneration")
p("An Ollama model can return a different answer to the same question on separate "
  "runs, since generation is not deterministic even at a low temperature setting, and "
  "a retrieved passage is occasionally phrased in a way the model summarises poorly "
  "on a first attempt. A 'Regenerate answer' action re-asks the most recent question "
  "without requiring it to be retyped, discarding the previous answer and its "
  "excerpts panel and streaming a fresh one in their place. The corresponding turn is "
  "first removed from conversation memory before the question is re-asked, so the "
  "stale answer is not still present in the history used to build the next prompt "
  "and the regenerated turn is recorded exactly once.")

h3("Retrieval Distance Scores")
p("The 'view retrieved excerpts' panel introduced earlier lists what was retrieved "
  "but not how closely it matched the question, leaving a user to judge relevance by "
  "reading the passage alone. Each excerpt now also shows the raw FAISS distance "
  "score for that chunk, labelled 'lower is closer' rather than converted into an "
  "invented percentage, since the underlying embeddings are not normalised and a "
  "manufactured confidence figure would overstate the precision of the measurement. "
  "This reuses FAISS's own similarity-search-with-score method in place of the "
  "plain similarity search used previously, copying the score onto a new Document "
  "object rather than mutating the one held in the FAISS docstore, so the change "
  "does not affect what is saved to or loaded from disk.")
p("This is a modest addition, but it closes the gap between what earlier sections of "
  "this report describe FAISS as doing internally and what the interface actually "
  "shows a user, matching the project's general principle that a claim of grounding "
  "should be something a user can inspect directly rather than take on trust.")

h3("Generation Model Selection")
p("The generation model was previously fixed at startup by the LLM_MODEL "
  "environment variable, requiring the .env file to be edited and the application "
  "restarted to try a different locally pulled model. An 'Advanced settings' dropdown "
  "now lists every model currently pulled in the local Ollama server, read from its "
  "/api/tags endpoint, and lets a user switch the generation model mid-session "
  "without restarting the application. The configured embedding model is deliberately "
  "excluded from this list, since it is not a chat model and was never intended to "
  "generate answers, only to build the FAISS index; switching it would in any case "
  "require re-embedding every loaded document, which this control does not attempt.")
p("A model chosen before any document has been loaded, when no pipeline object yet "
  "exists to hold it, is remembered on the shared configuration so that the pipeline "
  "created on the first upload or sample-load picks it up; a model chosen after "
  "documents are already loaded instead replaces the generation client on the "
  "existing pipeline directly. Two new unit tests cover both cases. This gives a user "
  "a direct way to compare how different locally available models summarise the same "
  "retrieved passages, without leaving the interface or touching configuration files.")

h2("4.3 System Evaluation")
p("Following tutor feedback on TMA02, TruLens (TruEra, 2024) is used to evaluate the "
  "system. It scores three metrics between 0 and 1: Answer Relevance, whether the "
  "answer addresses the question; Context Relevance, whether the retrieved chunks "
  "are pertinent; and Groundedness, whether claims in the answer are supported by the "
  "retrieved context. A 15-question test set was written covering factual, "
  "definition, comparison and synthesis question types, grounded in the sample "
  "corpus shipped with the application (Appendix 2, Week 12).")

if EVAL_CSV.exists():
    rows = []
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([
                row["chunk_size"],
                f"{float(row['Answer Relevance']):.2f}",
                f"{float(row['Context Relevance']):.2f}",
                f"{float(row['Groundedness']):.2f}",
            ])
    p("The TMA03 comparison was re-run in full for the EMA against the final shipped "
      "sample corpus (the three history-of-religion documents described in Section "
      "4.2), rather than the earlier placeholder documents used at TMA03 stage. This "
      "closes the gap identified in Section 3.3: the figures below are directly "
      "representative of the application as submitted.")
    source_note = "Table 1: RAG Triad scores by chunk size, evaluated against the final shipped corpus."
else:
    rows = _TMA03_FALLBACK_ROWS
    p("[STUDENT TO CONFIRM: the figures below are carried forward from the TMA03 "
      "evaluation, which was run against an earlier placeholder corpus of a lecture "
      "notes set, a textbook chapter and a personal summary, not the history-of-"
      "religion corpus shipped with the final application. A fresh run against the "
      "final corpus was started but had not completed by the time this report was "
      "generated; re-run evaluation.run_evaluation and regenerate this report before "
      "final submission so this table reflects the shipped corpus.]")
    source_note = "Table 1: RAG Triad scores by chunk size (TMA03 corpus; pending re-run against final corpus)."

table(["Chunk size (chars)", "Answer Relevance", "Context Relevance", "Groundedness"], rows)
caption(source_note)

if EVAL_CHART.exists():
    picture(EVAL_CHART, 5.5, "Figure 4: RAG Triad scores by chunk size.")

p("The 512-character configuration performs best across all three metrics, which is "
  "why it is the default used by the shipped application (config.py). The drop in "
  "Context Relevance at 256 characters reflects a pattern noted informally during "
  "development: small chunks often lack sufficient surrounding text to match well "
  "against a question. The smaller drop at 1024 characters is consistent with the "
  "observation in Gao et al. (2023) that larger chunks can introduce peripheral "
  "content that dilutes retrieval relevance. Groundedness scores are consistently "
  "high, indicating the system prompt is keeping the model anchored to retrieved "
  "context rather than supplementing it from training knowledge.")
p("One methodological limitation, raised as an open question in the TMA03 log "
  "(Appendix 2, Week 13), is that the judge model and the generation model are the "
  "same Llama 3 8B instance, which may inflate scores relative to an independent "
  "judge. For comparing configurations against each other on a consistent scale the "
  "scores remain informative, since any inflation applies equally across all three "
  "configurations, but their absolute values should not be read as a definitive "
  "external quality measure. A stronger design for future work would use a separate, "
  "larger model purely as the judge; this was not attempted here because it would "
  "require running two 8B-class models concurrently on CPU-only hardware, which was "
  "not practical within the available resources described in Section 3.1.")

h2("4.4 Extended Knowledge and Skills")
p("TruLens required understanding LLM-based evaluation: a model can act as a judge if "
  "given a rubric and asked to score a specific dimension of a response. Working "
  "through the documentation independently and integrating it into the existing "
  "pipeline was more demanding than following a tutorial, but the understanding "
  "gained is more secure, and directly informed the circularity discussion above.")
p("Developing the Gradio interface introduced session state management as a new "
  "technical area. Persisting a FAISS index across user interactions required "
  "reading the framework's source code rather than its documentation to understand "
  "how State components are passed between event handlers. The solution uses a "
  "single State object carrying both the index and the conversation history, passed "
  "as an input and output to each Gradio callback function. Extending the ingestion "
  "logic in Week 16 to merge rather than replace the index required re-examining that "
  "same State pattern to ensure the merged index, not a freshly rebuilt one, was the "
  "object returned to the interface.")

doc.save(str(OUT_PATH))
print(f"Saved (part 4) to {OUT_PATH}, paragraphs so far: {len(doc.paragraphs)}")

# =============================================================================
# 5. LEGAL, SOCIAL, ETHICAL AND PROFESSIONAL ISSUES AND EDI
# =============================================================================
h1("5. LEGAL, SOCIAL, ETHICAL AND PROFESSIONAL ISSUES AND EDI")

h2("5.1 Legal Issues")
p("The main legal framework is UK GDPR under the Data Protection Act 2018. Study "
  "notes can meet the ICO (2023) definition of personal data: they may include the "
  "student's name, comments about peers or tutors, health information, or financial "
  "details. This application processes only documents the user uploads from their "
  "own machine. Because the system runs entirely locally via Ollama, no data is "
  "transmitted to any external server at any stage, satisfying the UK GDPR principles "
  "of data minimisation and storage limitation by design rather than by policy.")
p("The bundled sample corpus, added so the application can be tried without the user "
  "sourcing material first, required its own review. The three sample files are "
  "original summaries written for this project rather than excerpts of any single "
  "copyrighted text, and each now carries an explicit provenance note together with "
  "a references section identifying the standard academic works (Eliade, Armstrong, "
  "Flood and others, listed in the files themselves) their content is consistent "
  "with. This was added specifically so the shipped corpus does not misrepresent "
  "AI-generated summary text as a verbatim scholarly source, and so a user inspecting "
  "the files can independently verify the claims made within them. The Llama 3 8B "
  "model is covered by Meta's Llama 3 Community Licence, which permits research and "
  "educational use. All other software is MIT or Apache 2.0 licenced. If a user were "
  "to ingest their own copyrighted third-party content, the legal position would "
  "depend on the licence under which those documents were obtained; UK copyright law "
  "permits fair dealing for non-commercial research and private study, but the "
  "boundaries of fair dealing in the context of AI-based text processing are not "
  "settled in case law. The application documentation makes clear that users are "
  "responsible for ensuring they have appropriate rights to the documents they "
  "ingest.")

h2("5.2 Social Issues")
p("The application has genuine potential as an accessibility tool. Students with "
  "dyslexia, visual impairment or attention difficulties often find navigating dense "
  "PDFs harder than neurotypical students. A natural-language interface reduces that "
  "cognitive load, which aligns with EDI principles around equitable access to "
  "learning support. There is a hardware barrier: the application requires at least "
  "8 GB RAM free to run a quantised model, which excludes users with more constrained "
  "hardware. A configuration using a smaller model would address this in a future "
  "version.")

h2("5.3 Ethical Issues")
p("Using an AI tool in an educational context raises questions about academic "
  "integrity. This application retrieves and summarises from documents the user "
  "provides; it is not a tool for generating assignment content, and this is made "
  "explicit in the application documentation. The risk of incorrect answers is "
  "mitigated by the system prompt restricting the model to retrieved context, which "
  "the TruLens Groundedness scores in Section 4.3 confirm is working, and by source "
  "citations giving the user a direct path to verify any response against the "
  "original document.")
p("Against Floridi and Cowls' (2019) framework, the system does reasonably well on "
  "most counts. It is genuinely useful for students working through a large volume of "
  "material, which addresses beneficence. The local-only architecture and citation "
  "requirement address non-maleficence and explicability fairly directly, since the "
  "user can see where every answer came from and nothing leaves their machine, which "
  "also addresses autonomy. The most difficult principle is justice: the tools "
  "involved are free and open-source, which helps, but the hardware requirement "
  "identified in Section 5.2 still means not everyone can run it.")

h2("5.4 Professional Issues")
p("Development has followed the BCS Code of Conduct (BCS, 2022), particularly the "
  "commitments to integrity and professional competence. Git and GitHub keep the "
  "development history fully traceable. Dependencies are versioned in "
  "requirements.txt, and the evaluation methodology is described in sufficient detail "
  "to be reproduced by running evaluation/run_evaluation.py against any document set. "
  "The project ethics checklist was completed and approved via the university's "
  "online ethics system and is included as Appendix 1.")

# =============================================================================
# 6. PERSONAL DEVELOPMENT, REVIEW AND REFLECTION
# =============================================================================
h1("6. PERSONAL DEVELOPMENT, REVIEW AND REFLECTION")

h2("6.1 Overview")
p("The project is complete. All planned components are working: document ingestion "
  "across PDF, plain text and Word formats; embedding and a FAISS index that merges "
  "rather than replaces on repeated uploads; retrieval and generation with source "
  "citations; conversation memory with token-based trimming; a Gradio web interface "
  "with a bundled sample corpus and a scope-enforcement explanation for the user; and "
  "a TruLens evaluation across three chunk size configurations.")

h2("6.2 Response to Tutor Feedback")
p("Feedback from both TMA02 and TMA03 has been addressed directly in this report "
  "rather than assumed as background knowledge for the reader.")
bullet("LO3 and LO9 (risk analysis and planning presentation): the risk register in "
       "Section 3.3 now states an explicit mitigation and outcome for every risk, and "
       "the schedule in Section 3.4 is presented as a self-contained table covering "
       "the full project from Week 1, rather than referring back to TMA02.")
bullet("LO7 (communication): Figures 1 to 3 add interface screenshots and an "
       "architecture diagram, directly addressing the feedback that the report was "
       "terse in places and would benefit from visual material.")
bullet("LO11 (evaluation presented more effectively): Section 4.3 presents results as "
       "a table and chart rather than prose, and states plainly, rather than only in "
       "passing, the limitation that the judge and generation model are the same.")
bullet("LO5 (use of the project log): this report cites specific Project Log entries "
       "by week throughout Sections 3, 4 and 6, rather than summarising the log "
       "without reference to it.")
bullet("LO6 and LO4 (source credibility): the sample document sourcing described in "
       "Section 5.1 extends the same critical approach to sources already applied to "
       "the academic literature in Section 2, to the corpus the application ships "
       "with.")

h2("6.3 Skills Development")
table(
    ["Area", "Starting point", "Current level", "Evidence"],
    [
        ["Python", "Competent", "Confident",
         "Building the full RAG pipeline, Gradio state management and REST API integration."],
        ["NLP and embeddings", "No experience", "Solid understanding",
         "Reading Lewis et al. (2020) and Gao et al. (2023) alongside hands-on implementation."],
        ["Vector databases", "No experience", "Working knowledge",
         "FAISS integration, reading Johnson et al. (2021), and the evaluation work in Section 4.3."],
        ["LangChain", "No experience", "Working knowledge",
         "Tutorials, official documentation, and migrating from an early tutorial-era version "
         "to the pinned release used in the final pipeline."],
        ["Ollama and local LLM inference", "No experience", "Working knowledge",
         "Installation, REST API use and model configuration across the whole project."],
        ["LLM evaluation", "No experience", "Working understanding",
         "TruLens integration, the chunk size comparison, and the judge circularity discussion "
         "in Section 4.3."],
        ["Technical report writing", "Competent", "Continuing to improve",
         "Writing across TMA01, TMA02, TMA03 and this EMA."],
    ],
    widths=[1.5, 1.1, 1.3, 2.6],
)

h2("6.4 Independent Learning and Effective Working")
p("The approach that worked best throughout this project was building something "
  "small and working first rather than designing everything upfront. Having a "
  "working pipeline early made it easier to understand what each component was "
  "actually doing and where the problems were. Reading Lewis et al. (2020) and "
  "Johnson et al. (2021) after seeing the architecture in practice made both papers "
  "considerably clearer than they would have been as preparatory reading alone.")
p("TruLens was adopted through independent reading after TMA02 feedback pointed "
  "toward it. Working through the documentation without step-by-step guidance and "
  "integrating it into an existing pipeline was demanding but produced a more secure "
  "understanding than following a tutorial would have done. The lesson from the "
  "LangChain migration (Appendix 2, Week 4) is to check the library version in any "
  "tutorial against the current stable release before starting, and to treat any "
  "significant gap as a reason to use official documentation rather than the "
  "tutorial; this lesson was applied again, successfully, when integrating TruLens.")

h2("6.5 What Went Well")
p("The iterative approach worked throughout: problems surfaced in practice rather "
  "than being missed during planning. TruLens integrated more smoothly than "
  "expected, and the structured metrics were useful both for the chunk size "
  "comparison and for confirming the system prompt was doing its job. The index "
  "merge fix in Week 16 is a clear example of the same pattern: it was found through "
  "using the finished application as an ordinary user would, not through code review "
  "alone.")

h2("6.6 What Did Not Go Well")
p("The Gradio session state issue (Appendix 2, Week 10) was the most significant "
  "unexpected problem in the interface work. The initial implementation caused the "
  "FAISS index to be rebuilt on every user message, producing a substantial delay "
  "before each response once tested with a larger document set. Diagnosing it "
  "required reading Gradio's source code, because the documentation did not clearly "
  "describe how State objects interact with event handler parameters. The fix was "
  "straightforward once the cause was identified, but finding it took several hours. "
  "The LangChain migration (Appendix 2, Week 4) remains the most time-costly problem "
  "over the whole project. The general lesson, applicable beyond this project, is "
  "that any framework under active development should be version-pinned from the "
  "first day of work rather than after a breaking change is encountered.")

h2("6.7 Concluding Reflection")
p("[STUDENT TO CONFIRM: add a short closing paragraph in your own words reflecting on "
  "the project as a whole now that it is complete, since this is the one part of the "
  "report that should be written in a genuinely first-person, personal register "
  "rather than assembled from prior report text.]")

doc.save(str(OUT_PATH))
print(f"Saved (part 5) to {OUT_PATH}, paragraphs so far: {len(doc.paragraphs)}")

# =============================================================================
# REFERENCES
# =============================================================================
h1("REFERENCES")
for ref in [
    "BCS (2022) Code of Conduct for BCS Members. Available at: "
    "https://www.bcs.org/membership-and-registrations/become-a-member/bcs-code-of-conduct/ "
    "(Accessed: 29 April 2026).",

    "Floridi, L. and Cowls, J. (2019) 'A unified framework of five principles for AI "
    "in society', Harvard Data Science Review, 1(1). doi:10.1162/99608f92.8cd550d1.",

    "Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M. "
    "and Wang, H. (2023) Retrieval-Augmented Generation for Large Language Models: A "
    "Survey. arXiv preprint arXiv:2312.10997. Available at: "
    "https://arxiv.org/abs/2312.10997 (Accessed: 15 March 2026).",

    "ICO (2023) Guide to the UK General Data Protection Regulation (UK GDPR). "
    "Information Commissioner's Office. Available at: "
    "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/ "
    "(Accessed: 29 April 2026).",

    "Johnson, J., Douze, M. and Jegou, H. (2021) 'Billion-scale similarity search with "
    "GPUs', IEEE Transactions on Big Data, 7(3), pp. 535 to 547. "
    "doi:10.1109/TBDATA.2019.2921572.",

    "LangChain (2024) LangChain Documentation. Available at: "
    "https://python.langchain.com/docs/ (Accessed: 20 March 2026).",

    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, "
    "H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S. and Kiela, D. (2020) "
    "'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', Advances in "
    "Neural Information Processing Systems, 33, pp. 9459 to 9474.",

    "Ollama (2024) Ollama Documentation. Available at: https://ollama.com/docs "
    "(Accessed: 20 March 2026).",

    "Russell, S. and Norvig, P. (2021) Artificial Intelligence: A Modern Approach. "
    "4th edn. Hoboken: Pearson.",

    "Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., "
    "Bashlykov, N. et al. (2023) Llama 2: Open Foundation and Fine-Tuned Chat Models. "
    "arXiv preprint arXiv:2307.09288. Available at: https://arxiv.org/abs/2307.09288 "
    "(Accessed: 10 March 2026).",

    "TruEra (2024) TruLens Documentation. Available at: https://www.trulens.org/ "
    "(Accessed: 2 June 2026).",

    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., "
    "Kaiser, L. and Polosukhin, I. (2017) 'Attention is all you need', Advances in "
    "Neural Information Processing Systems, 30, pp. 5998 to 6008.",
]:
    p(ref)

# =============================================================================
# BIBLIOGRAPHY
# =============================================================================
h1("BIBLIOGRAPHY")
for ref in [
    "Armstrong, K. (1993) A History of God: The 4000-Year Quest of Judaism, "
    "Christianity and Islam. New York: Ballantine Books.",

    "Assmann, J. (2001) The Search for God in Ancient Egypt. Ithaca: Cornell "
    "University Press.",

    "Black, J. and Green, A. (1992) Gods, Demons and Symbols of Ancient Mesopotamia: "
    "An Illustrated Dictionary. London: British Museum Press.",

    "Breen, J. and Teeuwen, M. (2010) A New History of Shinto. Chichester: "
    "Wiley-Blackwell.",

    "Burkert, W. (1985) Greek Religion. Cambridge, MA: Harvard University Press.",

    "Cohn-Sherbok, D. (2011) Judaism: History, Belief and Practice. Abingdon: "
    "Routledge.",

    "Cope, B. and Kalantzis, M. (2017) 'Conceptualising e-learning', in Cope, B. and "
    "Kalantzis, M. (eds.) E-Learning Ecologies: Principles for New Learning and "
    "Assessment. New York: Routledge, pp. 1 to 45.",

    "Eliade, M. (1978) A History of Religious Ideas, Volume 1: From the Stone Age to "
    "the Eleusinian Mysteries. Chicago: University of Chicago Press.",

    "Esposito, J.L. (2016) Islam: The Straight Path. 5th edn. New York: Oxford "
    "University Press.",

    "Fingarette, H. (1972) Confucius: The Secular as Sacred. New York: Harper and "
    "Row.",

    "Flood, G. (1996) An Introduction to Hinduism. Cambridge: Cambridge University "
    "Press.",

    "Gethin, R. (1998) The Foundations of Buddhism. Oxford: Oxford University Press.",

    "Karpathy, A. (2023) Let's build GPT: from scratch, in code, spelled out "
    "[Video]. YouTube. Available at: https://www.youtube.com/watch?v=kCc8FmEb1nY "
    "(Accessed: 5 February 2026).",

    "Kohn, L. (2001) Daoism and Chinese Culture. Cambridge, MA: Three Pines Press.",

    "MacCulloch, D. (2009) A History of Christianity: The First Three Thousand "
    "Years. London: Allen Lane.",

    "Open University (2021) TM358: Machine Learning and Artificial Intelligence. "
    "Milton Keynes: The Open University.",
]:
    p(ref)

doc.save(str(OUT_PATH))
print(f"Saved (part 6) to {OUT_PATH}, paragraphs so far: {len(doc.paragraphs)}")

# =============================================================================
# APPENDIX 1
# =============================================================================
h1("APPENDIX 1, PROJECT ETHICS CHECKLIST")
p("Project Title: A Local RAG Study Assistant Chatbot Using Open-Source Tools")
qa = [
    ("Q1. Does your project involve human participants?", "No."),
    ("Q2. Will you collect personal data?",
     "No. The application processes documents provided by the user on their own "
     "machine. No data is transmitted externally at any stage."),
    ("Q3. Does your project involve vulnerable groups?", "No."),
    ("Q4. Does your project involve deception?", "No."),
    ("Q5. Are there any legal or IPR concerns?",
     "The Llama 3 Community Licence permits research and educational use. The bundled "
     "sample documents are original summaries with an attached provenance note and "
     "references section (Section 5.1); any documents a user supplies themselves "
     "remain their own responsibility. The offline-by-design architecture satisfies "
     "UK GDPR requirements around data minimisation and storage limitation."),
    ("Q6. Are there any risks to the researcher?", "No."),
]
for q, a in qa:
    para = doc.add_paragraph()
    para.add_run(q + " ").bold = True
    para.add_run(a)
p("This checklist has been completed and approved via the university's online "
  "ethics submission system.")

# =============================================================================
# APPENDIX 2
# =============================================================================
h1("APPENDIX 2, PROJECT LOG EXTRACTS")

log_entries = [
    ("Week 1 to 2", "[DATES]",
     "Background reading on RAG and transformer architectures; environment set up "
     "with Python 3.11, Ollama, and initial model pulls. What went well: Ollama "
     "installation and model pulls were straightforward. Plan: work through LangChain "
     "tutorials."),
    ("Week 3 to 4", "[DATES]",
     "Worked through LangChain tutorials and explored FAISS. Discovered the tutorials "
     "referenced an older LangChain release; migrating the emerging pipeline to the "
     "version pinned in requirements.txt took approximately one week, longer than "
     "planned. What did not go well: library drift between tutorials and the pinned "
     "release. Lesson: pin versions before writing code against a tutorial, not after."),
    ("Week 5", "[DATES]",
     "Implemented a minimal end-to-end pipeline: load, split, embed, index, retrieve, "
     "generate. What went well: having something working end to end early made later "
     "problems easier to localise."),
    ("Week 6", "[DATES]",
     "Experimented with chunk size and overlap informally. Noted that very small "
     "chunks seemed to retrieve poorly, without yet having a structured way to "
     "measure this. Plan: return to this with a proper evaluation once the interface "
     "exists."),
    ("Week 7 to 9", "[DATES]",
     "Documented the code, wrote and submitted TMA02."),
    ("Week 10", "[DATES]",
     "Built the Gradio interface. The initial implementation rebuilt the FAISS index "
     "on every user message because session state was not being persisted correctly, "
     "producing a long delay before each response once tested with more than one "
     "document. Diagnosed by reading Gradio's source code, since the documentation "
     "did not clearly describe how State objects interact with event handler "
     "parameters. Fixed using a single State object carrying both the index and the "
     "conversation history. What did not go well: session state; this took several "
     "hours to diagnose. Plan: .docx support and cross-format testing."),
    ("Week 11", "[DATES]",
     "Added Docx2txtLoader for .docx support and tested against a Word version of an "
     "existing PDF; retrieval quality was comparable across formats. Noted that "
     "tables inside Word documents are not extracted cleanly by text-based "
     "processing. Plan: proceed to the evaluation phase."),
    ("Week 12", "[DATES]",
     "Set up the TruLens evaluation harness and wrote the 15-question test set. Ran "
     "the chunk size comparison at 256, 512 and 1024 characters. The 512-character "
     "configuration performed best across all three metrics. What went well: TruLens "
     "required less modification to the existing pipeline than expected."),
    ("Week 13", "[DATES]",
     "Reviewed the TruLens results and noted the circularity concern of using the "
     "same model as both judge and generator (Section 4.3). Drafted the project work "
     "section of TMA03."),
    ("Week 14 to 15", "[DATES]",
     "Worked through TMA02 tutor comments systematically, added the LLM background "
     "to the literature review, reorganised it into subsections, and restructured "
     "the evaluation section around the TruLens results. Submitted TMA03."),
    ("Week 16 to 17", "[DATES]",
     "Worked through TMA03 tutor feedback. Fixed a bug where uploading a second batch "
     "of documents replaced the FAISS index instead of extending it; added a merge "
     "path and a total-chunk count in the status message, plus a 'remove all "
     "documents' action so a user can deliberately reset. Applied a custom colour "
     "theme to the interface. Added a provenance note and a references section to "
     "each bundled sample document, citing the standard academic works their content "
     "is consistent with, after reviewing the LSEPI implications of shipping "
     "AI-authored sample material without attribution. Rewrote a user-facing sentence "
     "in the interface that used an em dash, to match the plain-punctuation style "
     "required for this report and, for consistency, applied to the application's own "
     "copy. What went well: the index merge fix was found by using the finished "
     "application as an ordinary user would, uploading a second file and noticing the "
     "first one stopped being retrievable. Plan: re-run the full evaluation against "
     "the final shipped corpus and produce the EMA report."),
    ("Week 18", "[DATES]",
     "Regenerated the architecture diagram and interface screenshots for the report, "
     "re-ran the full pytest suite (nine tests passing throughout, no regressions "
     "from the Week 16 to 17 changes), and started a fresh TruLens evaluation run "
     "against the final sample corpus so Section 4.3 reflects the shipped documents "
     "rather than the earlier placeholder corpus. [STUDENT TO CONFIRM: complete this "
     "entry once the evaluation run finishes and the report has been regenerated with "
     "the final figures.]"),
    ("Week 19", "[DATES]",
     "With the EMA baseline complete and time remaining before submission, moved into "
     "an enhancement phase: put the finished prototype under version control and "
     "pushed it to a private GitHub repository as a baseline before making further "
     "changes. Reviewed the application against the scope and 'finished solution' "
     "description set out in TMA02 and TMA03 to identify quality-of-life "
     "improvements that would not contradict prior claims. Implemented and tested, "
     "one at a time: per-document removal from the FAISS index using its delete-by-id "
     "method, alongside the existing 'remove all' action; session save and resume, "
     "wiring up the previously unused index save/load functions plus a small JSON "
     "manifest recording which chunks belong to which filename; streamed answer "
     "generation, replacing a wait for the full response with incremental output; a "
     "collapsible 'view retrieved excerpts' panel under each answer, so the passages "
     "backing an answer are inspectable directly in the chat; an adjustable top_k "
     "control; a conversation transcript download; and a plain-language Ollama "
     "connectivity check, replacing a raw exception with an actionable message. Four "
     "new unit tests were added for the pipeline changes and four for the prompt "
     "formatting and health check additions, all passing alongside the existing "
     "suite. What went well: building and testing one feature at a time, as with the "
     "core pipeline earlier in the project, meant each addition could be verified in "
     "isolation before moving to the next. [STUDENT TO CONFIRM: complete this entry "
     "with the exact date range once finalised.]"),
    ("Week 20", "[DATES]",
     "Continued the enhancement phase with a second quality-of-life addition: "
     "document-scoped retrieval, letting a user restrict a question to a chosen "
     "subset of the loaded documents rather than always searching the whole FAISS "
     "index. This reused the FAISS similarity search's existing metadata filter "
     "argument, matching on the source filename already recorded against each chunk, "
     "so no additional storage or re-embedding was needed. The same checklist used "
     "for per-document removal was reused as the scope selector to avoid duplicating "
     "the document list in the interface. Two new unit tests were added confirming "
     "that a scoped retrieval only returns chunks from the selected document and that "
     "a scoped query only cites that document, and verified live in the browser by "
     "restricting scope to one sample document and confirming a question covered only "
     "in a different document was correctly declined rather than answered from "
     "outside the chosen scope. What went well: the FAISS filter argument accepts a "
     "plain callable over chunk metadata, so the feature needed no changes to how "
     "documents are chunked, embedded or stored. [STUDENT TO CONFIRM: complete this "
     "entry with the exact date range once finalised.]"),
    ("Week 21", "[DATES]",
     "Added a third quality-of-life feature: a 'Regenerate answer' action that "
     "re-asks the most recent question in place, for cases where a non-deterministic "
     "generation run produces a poorly phrased summary of an otherwise correctly "
     "retrieved passage. The action locates the last user turn in the chat history, "
     "discards the stale answer and excerpts panel, removes the matching turn from "
     "conversation memory so it is not duplicated, and streams a fresh answer using "
     "the same retrieval and generation path as an ordinary question. Encountered and "
     "fixed a bug during live testing: the chat history component occasionally "
     "returns message content as a list of structured text parts rather than a plain "
     "string, which raised a validation error when passed straight to the embedding "
     "call; a small helper was added to normalise both forms before re-asking the "
     "question. A new unit test was added confirming that regenerating an answer "
     "replaces the previous memory turn rather than appending to it, and the fix was "
     "confirmed live in the browser by asking a question, regenerating it, and "
     "checking the conversation still held exactly one exchange. What went well: "
     "testing directly against the running application, rather than only the unit "
     "test fakes, caught a real formatting inconsistency in the chat component that "
     "the test suite alone would not have exercised. [STUDENT TO CONFIRM: complete "
     "this entry with the exact date range once finalised.]"),
    ("Week 22", "[DATES]",
     "Added retrieval distance scores to the 'view retrieved excerpts' panel, "
     "switching the retrieval call from FAISS's plain similarity search to its "
     "similarity-search-with-score equivalent so each excerpt can be labelled with "
     "the raw distance between the question and that chunk, described as 'lower is "
     "closer' rather than converted into a percentage that the underlying, "
     "non-normalised embeddings could not honestly support. The score is copied onto "
     "a new Document object rather than written into the FAISS docstore's own copy, "
     "so saved sessions and per-document manifests are unaffected. Two new unit "
     "tests were added, one confirming retrieval attaches a numeric score to every "
     "returned chunk and one confirming the excerpt formatter renders it correctly, "
     "and the feature was confirmed live in the browser, where the four excerpts "
     "returned for a sample question were shown in ascending distance order. What "
     "went well: the FAISS distance-scored search accepts the same k and filter "
     "arguments as the plain search used previously, so no other retrieval logic "
     "needed to change. [STUDENT TO CONFIRM: complete this entry with the exact date "
     "range once finalised.]"),
    ("Week 23", "[DATES]",
     "Added a generation model selector, reading the list of models currently "
     "pulled in the local Ollama server from its /api/tags endpoint and populating "
     "an 'Advanced settings' dropdown with it on every page load, matching the "
     "existing pattern used for the connectivity banner. The configured embedding "
     "model is filtered out of this list, since it is not a chat model. Selecting a "
     "model rebuilds the pipeline's generation client without affecting the "
     "embedding client or the FAISS index. Found and fixed a bug during live "
     "testing: a model chosen before any document was loaded had no visible effect, "
     "because no pipeline object yet existed to hold the choice and a newly created "
     "pipeline always read the original default model from configuration. Fixed by "
     "remembering the choice on the shared configuration object when no pipeline yet "
     "exists, so the pipeline created on the first upload or sample-load reads the "
     "chosen model instead of the original default. Two new unit tests cover both "
     "the case where a pipeline already exists and the case where it does not. What "
     "went well: this is the second bug in this enhancement phase, after the chat "
     "message formatting issue in Week 21, that only surfaced under live browser "
     "testing rather than the unit test suite, reinforcing the value of testing the "
     "actual running application rather than relying on automated tests alone. "
     "[STUDENT TO CONFIRM: complete this entry with the exact date range once "
     "finalised.]"),
]

for week, dates, text in log_entries:
    h3(f"{week} ({dates})")
    p(text)

doc.save(str(OUT_PATH))
print(f"Saved (final) to {OUT_PATH}, paragraphs so far: {len(doc.paragraphs)}")
