# LLM Coursework — Transformers, Fine-Tuning, RAG & Agents

Three connected projects from a graduate course on **LLM-driven software development**.
Together they trace one arc: from classic transformer fine-tuning, through prompt
engineering and retrieval, to parameter-efficient fine-tuning, RAG, and agent design.

> **Context.** These are course projects, not production systems. The course is about
> building software *with* LLMs, so the development workflow was itself AI-assisted —
> my work was the engineering decisions, experimental setup, and analysis of the
> results below. Each project ran end-to-end on real data and real GPUs; the numbers
> reported here are measured, not illustrative.

## Projects

| # | Project | Focus | Stack | Headline result |
|---|---------|-------|-------|-----------------|
| 1 | [Transformer Sentiment Analysis](01-transformer-sentiment-analysis/) | Encoder vs. decoder fine-tuning | PyTorch, HF Transformers, BERT, GPT-1 | BERT **0.923 F1** vs. GPT-1 **0.899 F1** on IMDb |
| 2 | [Prompt Engineering & RAG for MT](02-mt-prompting-rag/) | Three EN↔TR translation strategies | Qwen2.5-7B, vLLM, FAISS, COMET-22 | MAPS **0.805** > RAG 5-shot 0.792 > zero-shot 0.763 |
| 3 | [LoRA Fine-Tuning, RAG QA & Agents](03-llm-finetuning-rag-agents/) | QLoRA + RAG + agent design | QLoRA/PEFT, ChromaDB, bge-m3, TRL | LoRA en→tr **0.819** (+0.109 over zero-shot) |

## The thread

The three projects build on one another, and the later results reference the earlier ones:

1. **Project 1** establishes the basics: fine-tune both an encoder (BERT) and a
   decoder (GPT-1) for the *same* classification task and compare them head-to-head.
2. **Project 2** moves from training to *steering* a frozen 7B model on machine
   translation — zero-shot vs. a multi-aspect prompting pipeline (MAPS) vs. dynamic
   retrieval-augmented few-shot (RAG).
3. **Project 3** closes the loop: instead of only prompting the frozen model, it
   **fine-tunes** it (QLoRA) for the same EN↔TR task — and the fine-tuned adapter
   (en→tr COMET **0.819**) ends up **beating Project 2's best prompting pipeline**
   (MAPS, 0.805) in a single decoding pass. It then adds a RAG QA system over Turkish
   high-school history textbooks and a design for a multilingual Reflection + ReAct agent.

## Tech stack across the three

- **Models**: BERT, GPT-1, Qwen2.5-7B-Instruct
- **Fine-tuning**: full fine-tuning (Project 1), QLoRA / PEFT + TRL `SFTTrainer` (Project 3)
- **Serving / inference**: HF Transformers, vLLM, 4-bit (bitsandbytes / NF4)
- **Retrieval**: FAISS (Project 2), ChromaDB + BAAI/bge-m3 embeddings (Project 3)
- **Evaluation**: scikit-learn metrics (Project 1), COMET-22 `wmt22-comet-da` (Projects 2 & 3), accuracy on TurkishMMLU (Project 3)
- **Agents (design)**: ReAct + Reflexion, multilingual routing
- **Infra**: rented single-GPU pods (RTX 6000 Ada / RTX PRO 4500 Blackwell), reproducible pipeline scripts

## Repository layout

```
llm-coursework/
├── 01-transformer-sentiment-analysis/   # BERT vs GPT-1 on IMDb
├── 02-mt-prompting-rag/                  # zero-shot / MAPS / RAG for EN↔TR MT
└── 03-llm-finetuning-rag-agents/         # QLoRA MT + RAG QA + agent design
```

Each project has its own `README.md` (problem → approach → results) and a `docs/`
folder with the deeper technical write-up.

## Reproducibility & artifacts

Large, regenerable artifacts (trained LoRA adapter weights, full prediction dumps,
model caches) are **git-ignored** to keep the repo light — every one of them is
reproducible from the pipeline scripts and configs that *are* committed. Measured
result files (COMET scores, accuracy JSONs, summary tables) are kept in each
project's `results/` or `outputs/logs/` folder as evidence of the actual runs.

---

*Author: Ali Ekrem Yapıcı · [github.com/yapicialiekrem](https://github.com/yapicialiekrem)*
