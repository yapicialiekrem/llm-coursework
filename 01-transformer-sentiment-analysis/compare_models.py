"""
compare_models.py — Side-by-side comparison of BERT and GPT-1 results

Run this script after both train_bert.py and train_gpt.py have completed.
It reads the saved results.json files and prints a formatted comparison table.
"""

import json
import os


RESULTS = {
    "BERT" : "outputs/bert/results.json",
    "GPT-1": "outputs/gpt/results.json",
}

METRICS = ["accuracy", "precision", "recall", "f1", "loss"]


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            "Please run the corresponding training script first."
        )
    with open(path) as f:
        return json.load(f)


def print_comparison():
    data = {name: load_results(path) for name, path in RESULTS.items()}

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON — IMDb Sentiment Classification")
    print("=" * 60)
    print(f"  {'Metric':<14} {'BERT':>12} {'GPT-1':>12}  Winner")
    print("-" * 60)

    # ── Per-metric rows ───────────────────────────────────────────────────────
    for metric in METRICS:
        bert_val = data["BERT"]["test_metrics"].get(metric, "N/A")
        gpt_val  = data["GPT-1"]["test_metrics"].get(metric, "N/A")

        # For loss, lower is better; for all others, higher is better
        if isinstance(bert_val, float) and isinstance(gpt_val, float):
            if metric == "loss":
                winner = "BERT" if bert_val < gpt_val else "GPT-1"
            else:
                winner = "BERT" if bert_val > gpt_val else "GPT-1"
        else:
            winner = "—"

        print(f"  {metric.capitalize():<14} {str(bert_val):>12} {str(gpt_val):>12}  {winner}")

    print("=" * 60)

    # ── Training history summary ───────────────────────────────────────────────
    for model_name, result in data.items():
        print(f"\n  {model_name} — per-epoch validation F1:")
        for rec in result.get("history", []):
            epoch   = rec.get("epoch",   "?")
            val_f1  = rec.get("val_f1",  "?")
            val_acc = rec.get("val_accuracy", "?")
            print(f"    Epoch {epoch}  |  val_f1={val_f1}  val_acc={val_acc}")

    print()


if __name__ == "__main__":
    print_comparison()
