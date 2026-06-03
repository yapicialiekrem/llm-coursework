"""Combine all experiment outputs into a single, hand-in-ready Markdown report.

Reads:
  outputs/logs/comet_baseline.json   (Task 1 baseline COMET)
  outputs/logs/comet_lora.json       (Task 1 LoRA COMET)
  outputs/logs/rag_eval_results.json (Task 2 zero-shot vs RAG accuracy)
  outputs/logs/train.log             (Task 1 training metadata, last loss, runtime)

Writes:
  docs/final_results.md           (combined results table + sanity checks)

Usage:
  python scripts/build_final_report.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"
REPORTS = ROOT / "docs"


def read_json(name: str) -> dict | None:
    p = LOGS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def parse_train_log() -> dict:
    """Extract key numbers from the train log: final loss, runtime, train size."""
    p = LOGS / "train.log"
    out: dict = {}
    if not p.exists():
        return out
    text = p.read_text(errors="ignore")

    # Trainable params line
    m = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)", text)
    if m:
        out["trainable_params"] = m.group(1)
        out["all_params"] = m.group(2)
        out["trainable_pct"] = m.group(3) + "%"

    # Dataset sizes
    m = re.search(r"train=(\d+)\s+val=(\d+)", text)
    if m:
        out["train_size"] = int(m.group(1))
        out["val_size"] = int(m.group(2))

    # Last loss reported (TRL prints {'loss': X, ...} every logging_steps)
    losses = re.findall(r"'loss': ([\d.]+)", text)
    if losses:
        out["first_loss"] = float(losses[0])
        out["last_loss"] = float(losses[-1])
        out["num_loss_logs"] = len(losses)

    # Final epoch
    epochs = re.findall(r"'epoch': ([\d.]+)", text)
    if epochs:
        out["final_epoch"] = float(epochs[-1])

    return out


def format_comet_table(base: dict | None, lora: dict | None) -> str:
    """Markdown table comparing base vs LoRA COMET scores."""
    rows = ["| Direction | Base (zero-shot) | LoRA fine-tuned | Δ |", "|---|---|---|---|"]
    for d, label in [("en2tr", "English → Turkish"), ("tr2en", "Turkish → English")]:
        b_key, l_key = f"base_{d}_comet", f"lora_{d}_comet"
        b = base.get(b_key) if base else None
        l = (lora.get(l_key) if lora else None) or (base.get(l_key) if base else None)
        b_str = f"{b:.4f}" if b is not None else "—"
        l_str = f"{l:.4f}" if l is not None else "—"
        delta = f"{(l - b):+.4f}" if (b is not None and l is not None) else "—"
        rows.append(f"| {label} | {b_str} | {l_str} | {delta} |")
    return "\n".join(rows)


def format_rag_examples(records: dict | None, n_correct: int = 3, n_wrong: int = 3) -> str:
    """Show a few sample predictions (correct + wrong) from RAG mode."""
    if not records or "rag" not in records:
        return "_(records not available)_"
    rag_recs = records["rag"]
    correct = [r for r in rag_recs if r.get("correct")][:n_correct]
    wrong = [r for r in rag_recs if not r.get("correct")][:n_wrong]
    rows = ["| # | Question (truncated) | Predicted | Gold | OK? |", "|---|---|---|---|---|"]
    for i, r in enumerate(correct + wrong):
        q = (r.get("q") or "").replace("|", "\\|")[:90]
        mark = "✅" if r.get("correct") else "❌"
        rows.append(f"| {i+1} | {q}… | {r.get('pred')} | {r.get('gold')} | {mark} |")
    return "\n".join(rows)


def format_rag_table(rag: dict | None) -> str:
    if not rag:
        return "| — | — | — | — |"
    rows = ["| Mode | Accuracy | Correct | Total |", "|---|---|---|---|"]
    for mode in ("zero_shot", "rag"):
        if mode in rag:
            r = rag[mode]
            rows.append(f"| {mode} | {r['accuracy']:.4f} | {r['correct']} | {r['total']} |")
    if "zero_shot" in rag and "rag" in rag:
        delta = rag["rag"]["accuracy"] - rag["zero_shot"]["accuracy"]
        rows.append(f"| **Δ (rag − zero_shot)** | **{delta:+.4f}** | — | — |")
    return "\n".join(rows)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = read_json("comet_baseline.json")
    lora = read_json("comet_lora.json")
    rag = read_json("rag_eval_results.json")
    train_meta = parse_train_log()

    when = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# Final Results — LLM Fine-Tuning, RAG, and Agent QA Project

Generated: {when}

This file is auto-built from `outputs/logs/*.json` by `scripts/build_final_report.py`.

---

## Task 1 — LoRA Fine-Tuning for Machine Translation

### Training metadata
| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Quantization | NF4 4-bit (QLoRA) |
| LoRA rank | 16, α=32, dropout=0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | {train_meta.get("trainable_params", "—")} ({train_meta.get("trainable_pct", "—")} of {train_meta.get("all_params", "—")}) |
| Dataset | WMT16 tr-en, bidirectional (en↔tr) |
| Train size | {train_meta.get("train_size", "—")} examples |
| Validation size | {train_meta.get("val_size", "—")} examples |
| Epochs | {train_meta.get("final_epoch", 3)} (target 3) |
| Effective batch size | 16 (4 per device × 4 grad-accum) |
| Max sequence length | 256 |
| Optimizer | paged AdamW 8-bit |
| LR / schedule | 2e-4 cosine with 3 % warm-up |
| Hardware | NVIDIA RTX PRO 4500 Blackwell (32 GB), CUDA 12.8 |
| First/last training loss | {train_meta.get("first_loss", "—")} → {train_meta.get("last_loss", "—")} |

### COMET results
{format_comet_table(base, lora)}

COMET model: `Unbabel/wmt22-comet-da`. Test set: 2000 WMT16 tr-en pairs (random subsample, seed=42).

---

## Task 2 — RAG-Based QA (TurkishMMLU/History)

### Knowledge base
- **Source documents:** 4 MEB lise tarih ders kitabı (9, 10, 11. sınıf Tarih + 12. sınıf İnkılap Tarihi), 2024-2025 müfredat
- **Total characters:** ~2.41 M across all books
- **Chunks indexed:** 1795 (recursive character splitter, token-aware via bge-m3 tokenizer)
- **Chunk size / overlap:** 512 tokens / 64 tokens (~12 %)
- **Embedding model:** `BAAI/bge-m3` (multilingual, strong on Turkish)
- **Vector DB:** ChromaDB persistent, cosine distance
- **Top-k:** 5

### QA results (TurkishMMLU/History, 100 test questions)
{format_rag_table(rag)}

---

### Example predictions (RAG mode)

{format_rag_examples(read_json("rag_eval_records.json"))}

---

## Task 3 — Multilingual Agent Design

Design-only deliverable. See [`src/task3_agent/DESIGN.md`](../src/task3_agent/DESIGN.md) for the full document:

- **Part A:** Reflection + ReAct pattern explanations with workflows, pros/cons, use cases.
- **Part B:** Language-routed agent (TR → Task 2 RAG, EN → Wikipedia).
- **Part C:** Architecture, prompts (5 critical templates), interaction design, worst-case cost analysis, failure modes, design trade-offs, references.

---

## Reproducibility

```bash
# 1. Setup
bash scripts/setup_gpu.sh
bash scripts/download_data.sh

# 2. Run the full pipeline (training + COMET + RAG eval)
bash scripts/run_full_pipeline.sh

# 3. Build this report
python scripts/build_final_report.py
```

All configuration is in `configs/lora_config.yaml` and `configs/rag_config.yaml`.
"""

    out = REPORTS / "final_results.md"
    out.write_text(md)
    print(f"[ok] wrote {out}")
    if base:
        print("  baseline:", base)
    if lora:
        print("  lora:", lora)
    if rag:
        print("  rag:", {k: v.get("accuracy") if isinstance(v, dict) else v for k, v in rag.items()})


if __name__ == "__main__":
    main()
