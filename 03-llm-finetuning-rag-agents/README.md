# LoRA Fine-Tuning, RAG QA & Agent Design

Three components on one base model (`Qwen2.5-7B-Instruct`): **(1)** parameter-efficient
fine-tuning for EN↔TR translation, **(2)** a retrieval-augmented QA system over Turkish
high-school history textbooks, and **(3)** a design for a multilingual Reflection + ReAct
agent.

## Components

| # | Component | Type | Headline result |
|---|-----------|------|-----------------|
| 1 | **LoRA fine-tuning** (QLoRA, WMT16 EN↔TR) | Implementation | en→tr COMET **0.819** (+0.109 over zero-shot) |
| 2 | **RAG QA** (TurkishMMLU History, ChromaDB + bge-m3) | Implementation | accuracy **0.33 → 0.40** with retrieval |
| 3 | **Multilingual agent** (Reflection + ReAct) | Design | [`docs/agent_design.md`](docs/agent_design.md) |

---

## 1. LoRA fine-tuning for EN↔TR translation

Fine-tune `Qwen2.5-7B-Instruct` with **QLoRA** (NF4 4-bit base + LoRA adapters) on WMT16,
**bidirectionally** — one adapter learns both en→tr and tr→en.

- **Data**: WMT16 `tr-en`, 30K pairs subsampled (`seed=42`), each used in both directions
  → 60K training examples; 2K-pair test set shared with the baseline.
- **LoRA**: `r=16`, `α=32`, dropout 0.05, on attention **and** MLP modules
  (`q,k,v,o,gate,up,down`) — ~40M trainable params (~0.53% of 7.66B).
- **Training**: 3 epochs, effective batch 16, seq len 256, LR 2e-4 cosine, paged AdamW
  8-bit, bf16, gradient checkpointing. ~8h on one RTX PRO 4500 (32 GB, Blackwell).

### Results (COMET-22, n=2,000)

| Direction | Zero-shot baseline | + LoRA (ours) | Δ |
|-----------|:------------------:|:-------------:|:--:|
| **en→tr** | 0.7099 | **0.8188** | **+0.1090 (+15.4%)** |
| tr→en | 0.7864 | 0.8209 | +0.0345 (+4.4%) |

The big win is **en→tr** — the harder direction for an English-heavy pre-trained model.
Notably, our single-pass LoRA en→tr (**0.819**) **edges past the best prompting pipeline
from [Project 2](../02-mt-prompting-rag/)** (MAPS, 0.805), and the adapter also eliminates
the base model's habit of prefixing translations with *"İşte ... çevirisi:"*. Measured
numbers: [`outputs/logs/comet_lora.json`](outputs/logs/comet_lora.json). Methodology and
the LoRA theory write-up: [`docs/lora_results.md`](docs/lora_results.md),
[`docs/lora_theory.md`](docs/lora_theory.md).

## 2. RAG QA over Turkish history textbooks

Answer **TurkishMMLU History** multiple-choice questions by retrieving from four real MEB
high-school history textbooks (9–12th grade, 2024–25 curriculum).

- **Ingest**: `pdfplumber` text extraction (~2.4M chars) → token-aware recursive chunking
  (512 tok / 64 overlap) → **1,795 chunks**.
- **Embed / store**: `BAAI/bge-m3` (1024-dim, normalized) → **ChromaDB** persistent index
  (cosine, with source metadata).
- **Retrieve / answer**: top-5 semantic search per question, answered by the Task-1 LoRA
  model with letter parsing.

### Results (TurkishMMLU History, 100 questions)

| System | Accuracy |
|--------|:--------:|
| Zero-shot (no retrieval) | 0.33 |
| **RAG (top-5 from textbooks)** | **0.40** |

**+7 percentage points** from grounding the model in the textbook chunks. Measured:
[`outputs/logs/rag_eval_results.json`](outputs/logs/rag_eval_results.json). Methodology
(chunking trade-offs, retrieval, eval): [`docs/rag_results.md`](docs/rag_results.md).

## 3. Multilingual agent — Reflection + ReAct (design)

A **design-only** component (no runtime code): a language-aware agent that routes Turkish
questions to the Task-2 RAG retriever and English questions to the Wikipedia API, runs a
**ReAct** loop (Thought → Action → Observation, max 3 iterations), drafts an answer, then
**self-critiques (Reflexion)** and optionally revises once before finalizing. Full
diagram, prompts, and trade-off discussion: [`docs/agent_design.md`](docs/agent_design.md)
and [`src/task3_agent/DESIGN.md`](src/task3_agent/DESIGN.md).

```
Question
  → langdetect (LLM fallback)
  → [TR] RAG retriever  |  [EN] Wikipedia API
  → ReAct loop (≤3 iter): Thought → Action → Observation
  → draft → Reflexion critique → (revise once if needed) → final answer
```

---

## Run it (single-GPU pod)

```bash
bash scripts/setup_gpu.sh          # Blackwell-compatible torch cu128 + deps
bash scripts/download_data.sh      # MEB PDFs + HF dataset prefetch
python -m src.task1_lora_mt.evaluate_comet --config configs/lora_config.yaml --baseline-only
python -m src.task2_rag.build_index --config configs/rag_config.yaml    # ChromaDB index
bash scripts/run_full_pipeline.sh  # LoRA train → COMET → RAG eval (~7–8h)
python scripts/build_final_report.py
```

## Layout

```
src/task1_lora_mt/    # QLoRA training, inference, COMET eval
src/task2_rag/        # PDF → chunk → bge-m3 → ChromaDB → retrieve → answer
src/task3_agent/      # DESIGN.md (design-only)
configs/              # lora_config.yaml, rag_config.yaml
scripts/              # setup, data download, pipeline, report builder
outputs/logs/         # measured result JSONs (COMET + RAG accuracy)
docs/                 # technical write-ups (LoRA theory/results, RAG, agent, final results)
```

> The trained **LoRA adapter weights (~154 MB)** and model caches are git-ignored — they
> regenerate from `scripts/run_full_pipeline.sh` and the committed configs. The committed
> result JSONs in `outputs/logs/` are the evidence of the actual runs.

## Stack & references

QLoRA / PEFT · TRL `SFTTrainer` · bitsandbytes (NF4) · ChromaDB · BAAI/bge-m3 ·
COMET-22 · pdfplumber. Key papers: LoRA (Hu et al., 2021), QLoRA (Dettmers et al., 2023),
ReAct (Yao et al., 2022), Reflexion (Shinn et al., 2023), TurkishMMLU (Yüksel et al.,
2024), BGE-M3 (Chen et al., 2024).
