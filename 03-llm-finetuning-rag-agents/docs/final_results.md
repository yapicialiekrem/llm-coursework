# Final Results — LLM Fine-Tuning, RAG, and Agent QA Project

Generated: 2026-05-24 07:47 UTC

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
| Trainable params | 40,370,176 (0.5273% of 7,655,986,688) |
| Dataset | WMT16 tr-en, bidirectional (en↔tr) |
| Train size | 60000 examples |
| Validation size | 1000 examples |
| Epochs | 3.0 (target 3) |
| Effective batch size | 16 (4 per device × 4 grad-accum) |
| Max sequence length | 256 |
| Optimizer | paged AdamW 8-bit |
| LR / schedule | 2e-4 cosine with 3 % warm-up |
| Hardware | NVIDIA RTX PRO 4500 Blackwell (32 GB), CUDA 12.8 |
| First/last training loss | 2.9871 → 0.6524 |

### COMET results
| Direction | Base (zero-shot) | LoRA fine-tuned | Δ |
|---|---|---|---|
| English → Turkish | 0.7099 | 0.8188 | +0.1089 |
| Turkish → English | 0.7864 | 0.8209 | +0.0345 |

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
| Mode | Accuracy | Correct | Total |
|---|---|---|---|
| zero_shot | 0.3300 | 33 | 100 |
| rag | 0.4000 | 40 | 100 |
| **Δ (rag − zero_shot)** | **+0.0700** | — | — |

---

### Example predictions (RAG mode)

| # | Question (truncated) | Predicted | Gold | OK? |
|---|---|---|---|---|
| 1 | Malazgirt Savaşı’ndan sonra Anadolu’da kurulan ilk Türk beyliklerinin;  I. Türk-İslam … | E | E | ✅ |
| 2 | Malazgirt Savaşı sonrası Anadolu’ya sahip olan Türkler, siyasi, askerî ve demografik ac… | C | C | ✅ |
| 3 | Aşağıdakilerden hangisi Osmanlı Devleti’nin konargöçerleri iskân ederek (yerleştirer… | C | C | ✅ |
| 4 | Selçukluların Anadolu akınları öncesinde Bizans Anadolu’da hâkim durumda idi. Özellikl… | C | D | ❌ |
| 5 | Türkiye tarihi açısından “Miryokefalon Savaşı” önemli bir dönüm noktasıdır. Bu savas… | C | E | ❌ |
| 6 | Türkiye Selçuklu Devleti döneminde;  - Kervansaraylar yapılması, -   Ahi teşkilatının … | C | E | ❌ |

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
