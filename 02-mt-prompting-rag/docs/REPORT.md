# Technical Report — Prompt Engineering & RAG for EN↔TR Machine Translation

## 1. Objective

Hold the model and test set fixed and ask: **how much translation quality comes from
prompting strategy alone?** Three systems translate the full WMT16 EN↔TR test split with
`Qwen2.5-7B-Instruct` and are scored with COMET-22.

## 2. Setup

| | Value |
|--|-------|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Decoding | greedy (`temperature=0.0`, `top_p=1.0`, `max_new_tokens=256`) |
| Test set | WMT16 EN↔TR test, ~3,000 segments |
| Metric | COMET-22 (`Unbabel/wmt22-comet-da`) |
| Serving | vLLM (full runs) / HF Transformers + MPS (local smoke tests) |

Greedy decoding is used throughout so differences in score come from the *strategy*, not
sampling noise.

## 3. Systems

### 3.1 Zero-shot
A single instruction prompt ("Translate the following English sentence to Turkish.") and
a single decode. This is the reference point.

### 3.2 MAPS — Multi-Aspect Prompting and Selection
After He et al. (TACL 2024). Three batched **knowledge-mining** passes extract, from the
source sentence: (a) keywords, (b) topics, (c) demonstration translations. These are fed
back as guidance to produce **four candidate** translations, and a final **LLM-as-judge**
pass selects the best. Cost: ~5 LLM passes per segment.

### 3.3 RAG dynamic 5-shot
- **Index**: embed the **source side** of the WMT16 *training* set with
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, L2-normalize, store in a
  FAISS `IndexFlatIP` (cosine similarity). Each vector points back to its `(src, tgt)` pair.
- **Retrieval**: embed the test source, take the top-K nearest training sources.
- **Diversity**: an **MMR-lite** filter over-retrieves `3K` candidates and greedily keeps
  ones whose pairwise cosine to already-selected examples stays below `0.92` — this kills
  the "top-5 are five paraphrases of the query" failure mode common to dense retrieval over
  MT corpora.
- **Prompt**: the surviving `(src, tgt)` pairs become a dynamic 5-shot prompt.

## 4. Results

| System | COMET ↑ | Relative cost |
|--------|:-------:|:-------------:|
| **MAPS** | **0.8050** | ~5 passes/segment (~80–100 min) |
| RAG dynamic 5-shot | 0.7921 | 1 pass + retrieval (~15–20 min) |
| Zero-shot | 0.7629 | 1 pass (~15 min) |

(`results/summary.json`; distribution in `results/comet_histogram.png`.)

- **MAPS** is best: **+0.042 COMET** over zero-shot. Reasoning about the source —
  surfacing keywords, topic, and worked demonstrations — before committing to a
  translation produces the highest quality, at the price of a 5-pass pipeline.
- **RAG** captures most of the benefit (**+0.029**) far more cheaply: relevant in-context
  examples nudge the frozen model toward the right register and terminology with one
  decode plus a cheap FAISS lookup.
- Both confirm the expected ordering: **structured prompting / retrieval > naive
  zero-shot**, even with the weights untouched.

## 5. Why these choices

- **Qwen2.5-7B-Instruct** — competitive multilingual (incl. Turkish) coverage, a defined
  chat template, first-class vLLM support.
- **COMET over BLEU** — BLEU's n-gram matching is unreliable for Turkish's agglutinative
  morphology; COMET's learned, embedding-based scoring tracks human judgment much better.
- **MMR-lite threshold 0.92** — empirically drops ~5–15% of the top-15 neighbours as
  redundant while keeping genuinely useful exemplars, so the limited in-context budget
  isn't spent on duplicates.

## 6. Limitations

- COMET is reference-based; scores are bounded by WMT16's single reference per segment.
- MAPS's quality is real but expensive — 5× the inference of zero-shot — so it suits
  offline/batch translation more than interactive use.
- RAG's exemplar quality depends on the training-side corpus; out-of-domain inputs
  retrieve weaker neighbours and see smaller gains.

## 7. References

- He et al., 2024. *Exploring Human-Like Translation Strategy with Large Language Models
  (MAPS).* TACL.
- Rei et al., 2022. *COMET-22.* WMT22 metrics shared task.
