"""One-off script to generate the system architecture diagram for the EMA report.
Not part of the application; run manually and the output PNG is embedded in the report.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

BOX_STYLE = dict(boxstyle="round,pad=0.3", linewidth=1.4)

boxes = {
    "user": (1.0, 8.6, 2.4, 0.9, "User\n(browser)", "#e8f4ea", "#2f6b3a"),
    "ui": (4.3, 8.6, 3.0, 0.9, "Gradio UI\n(ui.py)", "#eaf7f5", "#1f6f66"),
    "ingest": (7.7, 8.6, 2.0, 0.9, "Ingestion\n(PDF / .txt / .docx)", "#fdf0dd", "#8a5a12"),
    "split": (7.7, 7.0, 2.0, 0.9, "Chunking\n512 / 64 chars", "#fdf0dd", "#8a5a12"),
    "embed": (7.7, 5.4, 2.0, 0.9, "Embeddings\nnomic-embed-text\n(Ollama)", "#fdf0dd", "#8a5a12"),
    "index": (4.3, 5.4, 3.0, 0.9, "FAISS Index\n(indexing.py)", "#eaf7f5", "#1f6f66"),
    "retrieve": (4.3, 3.8, 3.0, 0.9, "Retrieval\ntop-4, cosine similarity", "#eaf7f5", "#1f6f66"),
    "memory": (1.0, 3.8, 2.4, 0.9, "Conversation\nMemory (memory.py)", "#e8f4ea", "#2f6b3a"),
    "prompt": (4.3, 2.2, 3.0, 0.9, "Prompt Assembly\n(prompts.py)", "#eaf7f5", "#1f6f66"),
    "llm": (7.7, 2.2, 2.0, 0.9, "Llama 3 8B\n(Ollama, local)", "#fdf0dd", "#8a5a12"),
    "answer": (4.3, 0.6, 3.0, 0.9, "Answer + Source Citation", "#e8f4ea", "#2f6b3a"),
}

centers = {}
for key, (x, y, w, h, label, face, edge) in boxes.items():
    box = FancyBboxPatch((x, y), w, h, transform=ax.transData,
                          facecolor=face, edgecolor=edge, **BOX_STYLE)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, color="#1a1a1a")
    centers[key] = (x, y, w, h)


def arrow(a, b, style="-|>", rad=0.0):
    xa, ya, wa, ha = centers[a]
    xb, yb, wb, hb = centers[b]
    ax_, ay_ = xa + wa / 2, ya + ha / 2
    bx_, by_ = xb + wb / 2, yb + hb / 2
    start = _edge_point(ax_, ay_, wa, ha, bx_, by_)
    end = _edge_point(bx_, by_, wb, hb, ax_, ay_)
    patch = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}",
                             arrowstyle=style, mutation_scale=14, linewidth=1.2,
                             color="#444444", shrinkA=4, shrinkB=4)
    ax.add_patch(patch)


def _edge_point(cx, cy, w, h, ox, oy):
    """Point on the rectangle's border, on the line from its centre toward (ox, oy)."""
    dx, dy = ox - cx, oy - cy
    if dx == 0:
        scale = (h / 2) / abs(dy)
    elif dy == 0:
        scale = (w / 2) / abs(dx)
    else:
        scale = min((w / 2) / abs(dx), (h / 2) / abs(dy))
    return cx + dx * scale, cy + dy * scale


arrow("user", "ui")
arrow("ui", "ingest")
arrow("ingest", "split")
arrow("split", "embed")
arrow("embed", "index")
arrow("ui", "retrieve", rad=-0.15)
arrow("index", "retrieve")
arrow("retrieve", "prompt")
arrow("memory", "prompt")
arrow("prompt", "llm")
arrow("llm", "answer", rad=0.2)
arrow("answer", "ui", rad=0.35)
arrow("answer", "memory", rad=-0.2)

# Citations are built from the retrieved chunks' metadata, not from the LLM's own
# output, so retrieval also has a direct path to the answer. Routed as a right-angle
# step confined to the empty gap column between Memory and Prompt Assembly, so it
# never crosses another box.
retrieve_left = (centers["retrieve"][0], centers["retrieve"][1] + centers["retrieve"][3] / 2)
answer_left = (centers["answer"][0], centers["answer"][1] + centers["answer"][3] / 2)
GAP_X = 3.75
ax.plot([retrieve_left[0], GAP_X], [retrieve_left[1], retrieve_left[1]], color="#444444", linewidth=1.2)
ax.plot([GAP_X, GAP_X], [retrieve_left[1], answer_left[1]], color="#444444", linewidth=1.2)
gap_patch = FancyArrowPatch((GAP_X, answer_left[1]), answer_left,
                             arrowstyle="-|>", mutation_scale=14, linewidth=1.2,
                             color="#444444", shrinkA=0, shrinkB=4)
ax.add_patch(gap_patch)

ax.set_title("RAG Study Assistant: System Architecture", fontsize=13, fontweight="bold", pad=14)
fig.text(0.5, 0.02, "All components run locally; no document content or question leaves the machine.",
          ha="center", fontsize=8.5, style="italic", color="#444444")

plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("data/eval_results/architecture_diagram.png", dpi=200)
print("saved")
