"""One-off script to generate a UML-style sequence diagram for the EMA report,
illustrating the "ask a question" flow as a distinct view from the component-level
architecture diagram in architecture_diagram.png.

Not part of the application; run manually and the output PNG is embedded in the report.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ACTORS = ["User", "Gradio UI", "Pipeline", "FAISS Index", "Ollama\n(nomic-embed-text)", "Ollama\n(Llama 3 8B)"]
N = len(ACTORS)

fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(0, 14)
ax.axis("off")

X = {actor: i for i, actor in enumerate(ACTORS)}
LIFELINE_TOP = 13.2
LIFELINE_BOTTOM = 0.6

for actor in ACTORS:
    x = X[actor]
    box = FancyBboxPatch((x - 0.55, LIFELINE_TOP - 0.35), 1.1, 0.7,
                          boxstyle="round,pad=0.08", transform=ax.transData,
                          facecolor="#eaf7f5", edgecolor="#1f6f66", linewidth=1.3)
    ax.add_patch(box)
    ax.text(x, LIFELINE_TOP, actor, ha="center", va="center", fontsize=9, color="#1a1a1a")
    ax.plot([x, x], [LIFELINE_BOTTOM, LIFELINE_TOP - 0.35], linestyle="--", color="#9a9a9a", linewidth=1.0, zorder=0)


def message(src, dst, y, label, dashed=False, self_call=False):
    xs, xd = X[src], X[dst]
    style = "->" if dashed else "-|>"
    linestyle = "dashed" if dashed else "solid"
    if self_call:
        patch = FancyArrowPatch((xs, y), (xs + 0.01, y - 0.55), connectionstyle="arc3,rad=1.2",
                                 arrowstyle=style, mutation_scale=13, linewidth=1.1,
                                 color="#333333", linestyle=linestyle)
        ax.add_patch(patch)
        ax.text(xs + 0.2, y - 0.25, label, ha="left", va="center", fontsize=7.3, color="#1a1a1a")
        return
    patch = FancyArrowPatch((xs, y), (xd, y), arrowstyle=style, mutation_scale=13,
                             linewidth=1.1, color="#333333", linestyle=linestyle)
    ax.add_patch(patch)
    mid = (xs + xd) / 2
    ax.text(mid, y + 0.14, label, ha="center", va="bottom", fontsize=8.2, color="#1a1a1a")


y = 12.2
step = 0.95
message("User", "Gradio UI", y, "ask(question)"); y -= step
message("Gradio UI", "Pipeline", y, "answer(question, history)"); y -= step
message("Pipeline", "Ollama\n(nomic-embed-text)", y, "embed(question)"); y -= step
message("Ollama\n(nomic-embed-text)", "Pipeline", y, "question vector", dashed=True); y -= step
message("Pipeline", "FAISS Index", y, "similarity_search(vector, k=top_k)"); y -= step
message("FAISS Index", "Pipeline", y, "top-k chunks + scores", dashed=True); y -= step
message("Pipeline", "Pipeline", y, "assemble prompt\n(context, history, question)", self_call=True); y -= step
message("Pipeline", "Ollama\n(Llama 3 8B)", y, "generate(prompt), streamed"); y -= step
message("Ollama\n(Llama 3 8B)", "Pipeline", y, "answer tokens", dashed=True); y -= step
message("Pipeline", "Pipeline", y, "record turn in\nconversation memory", self_call=True); y -= step
message("Pipeline", "Gradio UI", y, "answer + citation + excerpts", dashed=True); y -= step
message("Gradio UI", "User", y, "render streamed answer", dashed=True)

ax.set_title("RAG Study Assistant: Sequence Diagram for \u2018Ask a Question\u2019",
             fontsize=13, fontweight="bold", pad=18)
fig.text(0.5, 0.015,
          "Dashed arrows are return values. Self-calls represent in-process steps within the pipeline.",
          ha="center", fontsize=8.5, style="italic", color="#444444")

plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("data/eval_results/sequence_diagram.png", dpi=200)
print("saved")
