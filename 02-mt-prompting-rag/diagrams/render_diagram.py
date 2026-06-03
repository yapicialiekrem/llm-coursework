"""Render the RAG-MT architecture diagram as a PNG.

Pure matplotlib so it works on any environment without graphviz.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def box(ax, xy, w, h, text, color="#E8F1FF", edge="#1F4E79", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, xy1, xy2, label=None, color="#333"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.4, color=color)
    ax.add_patch(a)
    if label:
        mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")


def main(out_path: str = "diagrams/rag_architecture.png"):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --- Offline indexing column (top) -------------------------------------
    ax.text(3.0, 7.6, "Offline — Indexing", fontsize=12, weight="bold",
            color="#1F4E79")
    box(ax, (0.3, 6.4), 2.4, 0.9, "WMT16 train\n(EN ↔ TR pairs)",
        color="#FFF6E5", edge="#8A6A1B")
    box(ax, (3.3, 6.4), 2.6, 0.9,
        "Light filter\n(length, dedup)", color="#FFF6E5", edge="#8A6A1B")
    box(ax, (6.5, 6.4), 2.8, 0.9,
        "Multilingual encoder\n(paraphrase-MiniLM)", color="#FFF6E5",
        edge="#8A6A1B")
    box(ax, (9.9, 6.4), 1.9, 0.9, "FAISS IndexFlatIP\n(cosine)",
        color="#FFF6E5", edge="#8A6A1B")

    arrow(ax, (2.7, 6.85), (3.3, 6.85))
    arrow(ax, (5.9, 6.85), (6.5, 6.85))
    arrow(ax, (9.3, 6.85), (9.9, 6.85))

    # --- Online retrieval & generation (bottom) ----------------------------
    ax.text(3.0, 5.4, "Online — Translation", fontsize=12, weight="bold",
            color="#1F4E79")

    box(ax, (0.3, 4.0), 2.4, 0.9, "Test source\nsentence",
        color="#E8F1FF", edge="#1F4E79")
    box(ax, (3.3, 4.0), 2.6, 0.9, "Embed query\n(same encoder)",
        color="#E8F1FF", edge="#1F4E79")
    box(ax, (6.5, 4.0), 2.8, 0.9,
        "FAISS top-K\n(K' = 3·K, K=5)", color="#E8F1FF", edge="#1F4E79")
    box(ax, (9.9, 4.0), 1.9, 0.9,
        "MMR filter\n(threshold 0.92)", color="#E8F1FF", edge="#1F4E79")

    arrow(ax, (2.7, 4.45), (3.3, 4.45))
    arrow(ax, (5.9, 4.45), (6.5, 4.45))
    arrow(ax, (9.3, 4.45), (9.9, 4.45))

    # offline -> online link
    arrow(ax, (10.85, 6.4), (10.85, 4.9), label="loaded")

    # MMR -> prompt builder
    box(ax, (4.5, 2.4), 3.4, 1.1,
        "5-shot prompt builder\n(retrieved (src,tgt) pairs\n+ query as final exemplar)",
        color="#E5F5E0", edge="#2C7A2C")
    arrow(ax, (10.85, 4.0), (7.9, 3.0))
    arrow(ax, (1.5, 4.0), (4.5, 3.0), label="query src")

    # LLM
    box(ax, (8.4, 2.4), 3.4, 1.1,
        "Qwen 2.5 7B Instruct\n(vLLM, batched)", color="#F4E6FF",
        edge="#5E3A87")
    arrow(ax, (7.9, 2.95), (8.4, 2.95))

    # Output
    box(ax, (5.5, 0.6), 4.0, 1.0,
        "Final translation\n(Turkish ↔ English)", color="#FFEFEF",
        edge="#A02929")
    arrow(ax, (10.0, 2.4), (8.5, 1.6))

    # Title
    fig.suptitle("RAG-Based Machine Translation Architecture\n"
                 "Dynamic 5-shot example selection over WMT16 EN↔TR",
                 fontsize=13, weight="bold", color="#1F4E79")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved diagram -> {out_path}")


if __name__ == "__main__":
    main()
