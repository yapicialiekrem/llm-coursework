"""Render the per-segment COMET-22 histogram from results/comet_*.json.

Run after `scripts/05_evaluate.py` has produced the per-system score files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SYSTEMS = [
    ("zero_shot", "Zero-shot",  "#7a90b5"),
    ("rag_5shot", "RAG-5-shot", "#d09a3e"),
    ("maps",      "MAPS",       "#2c7a2c"),
]


def main():
    scores = {}
    for key, _, _ in SYSTEMS:
        path = Path(f"results/comet_{key}.json")
        scores[key] = json.loads(path.read_text())["segment_scores"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.3, 1.0, 40)
    for key, label, color in SYSTEMS:
        vals = scores[key]
        mean = sum(vals) / len(vals)
        ax.hist(vals, bins=bins, alpha=0.55, color=color,
                label=f"{label} (mean = {mean:.4f})", edgecolor="white",
                linewidth=0.3)
    ax.set_xlabel("Segment COMET-22 score")
    ax.set_ylabel("Number of segments")
    ax.set_title("Per-segment COMET-22 distribution — WMT16 EN→TR (n = 3 000)")
    ax.set_xlim(0.3, 1.0)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    out = "results/comet_histogram.png"
    plt.savefig(out, dpi=160)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
