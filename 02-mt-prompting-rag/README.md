# Prompt Engineering & RAG for Machine Translation

Three ways to translate **English ↔ Turkish** with a *frozen* 7B model — no fine-tuning,
only how you prompt and what context you retrieve — benchmarked head-to-head with COMET.

## Problem

Translate the **full WMT16 EN↔TR test split (~3,000 segments)** with
**Qwen2.5-7B-Instruct**, and measure how much translation quality you can buy through
prompting strategy alone. All three systems use the same model and the same test set; only
the prompting / retrieval differs. Quality is scored with **COMET-22**
(`Unbabel/wmt22-comet-da`), a learned metric that correlates with human judgment far
better than BLEU for a morphologically rich target like Turkish.

## The three systems

1. **Zero-shot** — one prompt, one decode. The baseline.
2. **MAPS** (Multi-Aspect Prompting and Selection, He et al., TACL 2024) — three batched
   knowledge-mining passes (keywords, topics, demonstrations), four candidate
   translations, then one LLM-as-judge selection pass.
3. **RAG dynamic 5-shot** — a FAISS index over the WMT16 *training* side + a multilingual
   sentence encoder retrieves the 5 most similar source sentences as in-context examples,
   with an MMR-lite diversity filter to avoid five near-duplicate neighbours.

## Results (WMT16 EN↔TR test, n=3,000, COMET-22)

| System | COMET ↑ |
|--------|:-------:|
| **MAPS** | **0.8050** |
| RAG dynamic 5-shot | 0.7921 |
| Zero-shot | 0.7629 |

**Takeaway.** Both strategies beat zero-shot. **MAPS wins (+0.042 COMET over zero-shot)**
by reasoning about the source before translating — but it pays for it with a 5-pass
pipeline (~80–100 min vs. ~15 min for zero-shot). RAG gets most of the gain (+0.029) at a
fraction of the cost by simply showing the model similar examples. Scores are in
[`results/summary.json`](results/summary.json); the per-system distribution is plotted in
[`results/comet_histogram.png`](results/comet_histogram.png).

> Context for the wider repo: in [Project 3](../03-llm-finetuning-rag-agents/), a LoRA
> fine-tune of the same Qwen2.5-7B reaches **0.819** en→tr in a *single* decode — edging
> past MAPS's 0.805. Prompting takes you far; fine-tuning takes you a bit further.

## Why these choices

- **Qwen2.5-7B-Instruct** — strong multilingual coverage including Turkish, a well-defined
  chat template, and first-class vLLM support.
- **COMET-22 only** — BLEU is unreliable on Turkish morphology; COMET is the metric the
  task targets. (BLEU is wired up in the notebook as a cheap cross-check.)
- **MMR-lite on RAG** — dense retrieval over an MT corpus tends to return five paraphrases
  of the query. A 0.92 cosine cap drops the redundant neighbours and keeps the in-context
  budget productive.

## Run it

The whole pipeline is also in [`mt_prompting_rag.ipynb`](mt_prompting_rag.ipynb). Via the
CLI scripts (the full run is ~3h, ~$5 on a rented RTX 6000 Ada):

```bash
pip install -r requirements.txt
python scripts/01_build_rag_index.py --direction en-tr   # FAISS index (one-off)
python scripts/02_run_zeroshot.py                        # ~15 min
python scripts/03_run_maps.py                            # ~80–100 min
python scripts/04_run_rag.py                             # ~15–20 min
python scripts/05_evaluate.py --pred outputs/preds_zeroshot.jsonl \
                              --pred outputs/preds_maps.jsonl \
                              --pred outputs/preds_rag.jsonl    # COMET scoring
```

Add `--limit 16` to any run script (or set `CONFIG['limit'] = 16` in the notebook) for a
fast smoke test before committing to the full run. On Apple Silicon, the
`transformers` + MPS backend with `Qwen2.5-3B-Instruct` runs the plumbing locally.

## Layout

```
mt_prompting_rag.ipynb      # end-to-end notebook
src/
  data.py                   # WMT16 loader, normalization, filters
  prompts.py                # prompt templates (zero-shot, MAPS, RAG)
  translator.py             # vLLM / transformers backends + pipelines
  rag.py                    # FAISS index + MMR-lite selection
  evaluate.py               # COMET-22 wrapper, JSONL I/O
scripts/                    # 01–05: build index → run systems → evaluate
results/                    # COMET scores + summary + histogram
diagrams/rag_architecture.png
docs/REPORT.md              # fuller write-up
```

> The full ~3,000-line prediction dumps (`outputs/preds_*.jsonl`) are git-ignored to keep
> the repo light; the COMET scores derived from them live in `results/`.

## Notes & limitations

- COMET is reference-based, so the scores depend on WMT16's single reference per segment.
- MAPS's quality comes at a real latency/cost premium (5 LLM passes per segment) — worth
  it for offline quality, less so for interactive use.
- Retrieval quality is bounded by the training-side corpus; out-of-domain inputs see
  weaker few-shot neighbours.
