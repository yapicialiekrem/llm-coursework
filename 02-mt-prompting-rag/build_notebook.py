"""Generate mt_prompting_rag.ipynb. Run once whenever the notebook content changes."""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# Prompt Engineering & RAG for Machine Translation\n\n"
        "**Author:** Ali Ekrem Yapıcı  \n"
        "**Course:** LLM Driven Software Development  \n"
        "**Due:** 21 / 05 / 2026  \n"
        "**Language pair:** English ↔ Turkish (WMT16)  \n"
        "**Model:** Qwen 2.5 7B Instruct (vLLM, RTX 6000 Ada 48 GB)  \n\n"
        "This notebook is the runnable companion to `report.pdf`. The report contains\n"
        "the literature review and theoretical discussion; this notebook contains the\n"
        "code, experiments, and final COMET-22 numbers.\n\n"
        "**How to use.** The notebook is laid out so that the heavy work (model\n"
        "loading, full-test-set generation, COMET scoring) can be triggered cell by\n"
        "cell on a rented RTX 6000 Ada box, but smoke-tested on a Mac with\n"
        "`limit = 16` and a smaller model. See `README.md` for the RTX 6000\n"
        "deployment recipe.\n"
    ),
    md(
        "## 0. Configuration\n\n"
        "All knobs live in one cell so the notebook can be re-run with different\n"
        "models, subset sizes, or batch sizes without hunting through the code."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "# Make `src` importable when the notebook is opened from the project root.\n"
        "sys.path.insert(0, str(Path.cwd()))\n"
        "\n"
        "CONFIG = {\n"
        "    # Direction we evaluate. The project requires EN↔TR — we run EN→TR by\n"
        "    # default; flip to 'tr-en' to reproduce the reverse direction.\n"
        "    'direction': 'en-tr',\n"
        "\n"
        "    # 0 = full WMT16 EN-TR test split (~3000 segments). This is what the\n"
        "    # submitted COMET numbers in report.pdf are based on. Set to a small\n"
        "    # number (e.g. 16) to smoke-test the pipeline locally on Mac.\n"
        "    'limit': 0,\n"
        "\n"
        "    # Model + backend. 'vllm' is the path used for the submitted run\n"
        "    # (RTX 6000 48 GB / H100). 'transformers' is the slow fallback.\n"
        "    'model_name': 'Qwen/Qwen2.5-7B-Instruct',\n"
        "    'backend':    'vllm',\n"
        "    'dtype':      'bfloat16',\n"
        "\n"
        "    # Retrieval.\n"
        "    'encoder_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',\n"
        "    'rag_k': 5,\n"
        "    'rag_max_index': 200_000,\n"
        "    'rag_mmr_threshold': 0.92,\n"
        "\n"
        "    # COMET-22 (reference-based).\n"
        "    'comet_model': 'Unbabel/wmt22-comet-da',\n"
        "    'comet_batch_size': 16,\n"
        "    'comet_gpus': 1,\n"
        "\n"
        "    # I/O.\n"
        "    'outputs_dir': 'outputs',\n"
        "    'results_dir': 'results',\n"
        "}\n"
        "Path(CONFIG['outputs_dir']).mkdir(exist_ok=True)\n"
        "Path(CONFIG['results_dir']).mkdir(exist_ok=True)\n"
        "CONFIG\n"
    ),
    md(
        "## Part 1 — Dataset Selection and Preparation\n\n"
        "We use the **WMT16** parallel corpus, configuration `tr-en` (the only TR\n"
        "config in the WMT16 shared task). It bundles SETIMES2 and a small portion\n"
        "of OpenSubtitles; the test split is the WMT16 newstest 2016 release.\n\n"
        "**Preprocessing pipeline (Part 1.3.c).**\n\n"
        "1. Download via the HuggingFace `datasets` builder `wmt16` / `tr-en`.\n"
        "2. Project each example into a flat `{src, tgt, src_lang, tgt_lang}` row\n"
        "   so the same pipeline serves both directions.\n"
        "3. Apply Unicode NFC normalization and whitespace collapsing — light,\n"
        "   reversible cleanup that preserves casing and diacritics so COMET\n"
        "   reflects translation quality rather than tokenization artifacts.\n"
        "4. For the **training-side RAG index only**, drop empty / very short / very\n"
        "   long pairs (10–400 chars). The **test split is left untouched**, as\n"
        "   required by the task.\n"
    ),
    code(
        "from src.data import load_wmt16, dataset_stats, sample_subset, filter_by_length\n"
        "\n"
        "ds = load_wmt16(CONFIG['direction'])\n"
        "stats = dataset_stats(ds)\n"
        "stats\n"
    ),
    code(
        "test = ds['test']\n"
        "if CONFIG['limit']:\n"
        "    test = sample_subset(test, CONFIG['limit'])\n"
        "items = [dict(r) for r in test]\n"
        "print(f'Direction : {CONFIG[\"direction\"]}')\n"
        "print(f'Test pairs: {len(items)}')\n"
        "print()\n"
        "print('--- Example input-output samples (Part 1.3.d) ---')\n"
        "for r in items[:3]:\n"
        "    print(f\"SRC ({r['src_lang']}): {r['src']}\")\n"
        "    print(f\"TGT ({r['tgt_lang']}): {r['tgt']}\")\n"
        "    print()\n"
    ),
    md(
        "### Dataset characteristics (Part 1.3.a–b)\n\n"
        "| Split | Pairs | Avg src tokens | Avg tgt tokens |\n"
        "|-------|------:|---------------:|---------------:|\n"
        "| train | ~205k | medium         | medium         |\n"
        "| validation | ~1000 | medium    | medium         |\n"
        "| test  | ~3000 | medium         | medium         |\n\n"
        "*(Exact numbers are printed by the `stats` cell above; keep that output\n"
        "in the submitted notebook so the table is grounded in the real dataset\n"
        "rather than these approximate counts.)*\n"
    ),
    md(
        "## Part 2 — Model Selection\n\n"
        "We pick **Qwen 2.5 7B Instruct** (`Qwen/Qwen2.5-7B-Instruct`).\n\n"
        "**Hardware justification (Part 2.2.a).** At bfloat16 the model is ~14 GB;\n"
        "with vLLM's PagedAttention and a 4 K context it fits comfortably in a\n"
        "single RTX 6000 Ada 48 GB and leaves ~25 GB of KV-cache headroom for\n"
        "large\n"
        "batches. On Apple Silicon (M-series), a 4-bit GGUF quantization runs at\n"
        "~10 tok/s, which is enough for the local smoke test but not for the full\n"
        "3000-sentence run.\n\n"
        "**MT suitability (Part 2.2.b).** Qwen 2.5 was pretrained on a corpus with\n"
        "substantial multilingual coverage (including Turkish) and a strong\n"
        "instruction-following post-training stage. On the public WMT24 results\n"
        "page the Qwen 2.5 7B baseline is competitive with Llama-3 8B Instruct\n"
        "and clearly above Mistral 7B Instruct v0.3 on EN↔non-English directions,\n"
        "particularly on lower-resource targets like Turkish. The model also has a\n"
        "well-defined chat template (`<|im_start|>` / `<|im_end|>`) that we rely on\n"
        "in `src/prompts.py`.\n\n"
        "**Experimental environment (Part 2.3).** Fill in the actual numbers from\n"
        "the run before submitting:\n\n"
        "- GPU: NVIDIA RTX 6000 Ada — 48 GB GDDR6 (rented via RunPod)\n"
        "- CPU: 16-core host (vCPU)\n"
        "- RAM: 128 GB host RAM (~22 GB peak for the COMET model)\n"
        "- Software: PyTorch 2.3, vLLM ≥ 0.6, transformers ≥ 4.45, FAISS-CPU 1.8,\n"
        "  sentence-transformers ≥ 3.0, `unbabel-comet` 2.2.\n"
    ),
    code(
        "from src.translator import GenConfig, VLLMBackend, TransformersBackend\n"
        "\n"
        "# Note: this cell allocates the model weights on the GPU and takes ~60 s on\n"
        "# RTX 6000 Ada. Run it once per session.\n"
        "if CONFIG['backend'] == 'vllm':\n"
        "    backend = VLLMBackend(model_name=CONFIG['model_name'], dtype=CONFIG['dtype'])\n"
        "else:\n"
        "    backend = TransformersBackend(model_name=CONFIG['model_name'], dtype=CONFIG['dtype'])\n"
        "\n"
        "gen_cfg   = GenConfig(max_new_tokens=256, temperature=0.0)\n"
        "judge_cfg = GenConfig(max_new_tokens=4,   temperature=0.0)\n"
        "backend\n"
    ),
    md(
        "## Part 3 — Prompt Engineering for Machine Translation\n\n"
        "### 3.A Literature review (see `report.pdf` §3)\n\n"
        "The paper *Exploring Human-Like Translation Strategy with Large Language\n"
        "Models* (He et al., TACL 2024; arXiv:2305.04118) introduces **MAPS** —\n"
        "*Multi-Aspect Prompting and Selection*. The proposed pipeline mimics how a\n"
        "human translator works: (1) **mine** three kinds of translation-relevant\n"
        "knowledge (keywords, topic, an analogous demonstration pair), (2)\n"
        "**generate** one knowledge-conditioned translation per aspect plus a\n"
        "vanilla baseline (four candidates total), and (3) **select** the best\n"
        "candidate either with an external QE scorer (Comet-QE) or, in the fully\n"
        "open-source variant, with an LLM-as-judge single-choice question\n"
        "(LLM-SCQ). Across 11 WMT22 directions and 3 LLMs, MAPS improves COMET-22\n"
        "by **0.5 – 1.5 points** over zero-shot and reduces hallucination /\n"
        "omission errors in MQM human evaluation. The full summary, including the\n"
        "exact prompt templates we reproduce here, is in §3 of the report.\n\n"
        "### 3.B Prompting patterns\n\n"
        "**Break Complex Tasks into Simpler Subtasks.** Instead of asking the LLM\n"
        "to simultaneously parse domain, recall terminology, mimic target-language\n"
        "style and write a fluent rendering in one shot, MAPS carves the problem\n"
        "into narrower sub-prompts (mine keywords → mine topic → mine demo →\n"
        "translate-with-knowledge → select). Each sub-prompt is easier to satisfy,\n"
        "and the selector filters out cases where a particular sub-prompt\n"
        "produced bad knowledge.\n\n"
        "**LLM as a Judge.** The same model that produced the four candidates is\n"
        "asked which one is best via a multiple-choice prompt that returns a\n"
        "single letter. No external evaluator is required — useful when we want\n"
        "to ship a fully open-source pipeline — but it inherits the model's\n"
        "biases (length, fluency, source-language style).\n\n"
        "### 3.C COMET vs BLEU (see `report.pdf` §3.C)\n\n"
        "**COMET** is a neural, embedding-based MT evaluation metric trained on\n"
        "human direct-assessment scores. We use `Unbabel/wmt22-comet-da`, the\n"
        "WMT22 reference-based checkpoint. It encodes (source, hypothesis,\n"
        "reference) with XLM-R and regresses to a quality score that correlates\n"
        "with human judgement substantially better than BLEU — BLEU is a surface\n"
        "n-gram overlap with the reference and is known to mis-rank semantically\n"
        "equivalent paraphrases, which is exactly the regime LLM translations\n"
        "operate in.\n"
    ),
    md(
        "### 3.D.1 Inspect a MAPS run on one example\n\n"
        "Before launching the full evaluation, the next cell runs the entire MAPS\n"
        "pipeline on a single sentence and prints every intermediate artefact —\n"
        "the three knowledge slots, the four candidates, the judge's choice. This\n"
        "is a useful sanity check and gives the reader a concrete picture of\n"
        "what the pipeline actually produces."
    ),
    code(
        "from src.translator import maps_translate\n"
        "\n"
        "sample = [items[0]]\n"
        "results = maps_translate(backend, sample, gen_cfg, judge_cfg)\n"
        "r = results[0]\n"
        "\n"
        "print('=== MAPS trace ===')\n"
        "print(f'Source     : {r.src}')\n"
        "print(f'Reference  : {sample[0][\"tgt\"]}')\n"
        "print()\n"
        "print(f'Keywords   : {r.intermediate[\"keywords\"]}')\n"
        "print(f'Topic      : {r.intermediate[\"topics\"]}')\n"
        "print(f'Demo       : {r.intermediate[\"demo\"]}')\n"
        "print()\n"
        "for i, c in enumerate(r.intermediate['candidates']):\n"
        "    label = ['vanilla','+kw','+topic','+demo'][i]\n"
        "    mark = '★' if i == r.intermediate['chosen'] else ' '\n"
        "    print(f'  {mark} ({label}) {c}')\n"
        "print()\n"
        "print(f'Judge raw  : {r.intermediate[\"judge_raw\"]!r}')\n"
        "print(f'Final      : {r.hypothesis}')\n"
    ),
    md(
        "### 3.D.2 Full-test-set generation: zero-shot vs. MAPS\n\n"
        "These two cells produce the prediction files `outputs/preds_zeroshot.jsonl`\n"
        "and `outputs/preds_maps.jsonl`. On RTX 6000 Ada with vLLM and the\n"
        "full 3 000-segment test set, the zero-shot pass takes ~15 min; MAPS\n"
        "takes ~80–100 min because it issues seven LLM calls per source\n"
        "(3 knowledge + 4 candidates + 1 judge — all batched).\n"
    ),
    code(
        "from tqdm.auto import tqdm\n"
        "from src.translator import zero_shot_translate\n"
        "from src.evaluate import save_predictions\n"
        "\n"
        "BATCH = 64\n"
        "hyps_zs = []\n"
        "for s in tqdm(range(0, len(items), BATCH), desc='zero-shot'):\n"
        "    batch = items[s:s + BATCH]\n"
        "    hyps_zs.extend(r.hypothesis for r in zero_shot_translate(backend, batch, gen_cfg))\n"
        "\n"
        "save_predictions(f\"{CONFIG['outputs_dir']}/preds_zeroshot.jsonl\",\n"
        "                 items, hyps_zs, extra={'system': 'zero_shot'})\n"
        "print('Saved', len(hyps_zs), 'zero-shot predictions')\n"
    ),
    code(
        "hyps_maps = []\n"
        "debug_maps = []\n"
        "for s in tqdm(range(0, len(items), BATCH), desc='MAPS'):\n"
        "    batch = items[s:s + BATCH]\n"
        "    res = maps_translate(backend, batch, gen_cfg, judge_cfg)\n"
        "    for it, r in zip(batch, res):\n"
        "        hyps_maps.append(r.hypothesis)\n"
        "        debug_maps.append({'src': it['src'], 'tgt': it['tgt'],\n"
        "                            'hypothesis': r.hypothesis, **r.intermediate})\n"
        "save_predictions(f\"{CONFIG['outputs_dir']}/preds_maps.jsonl\",\n"
        "                 items, hyps_maps, extra={'system': 'maps'})\n"
        "import json\n"
        "with open(f\"{CONFIG['outputs_dir']}/preds_maps_debug.jsonl\", 'w', encoding='utf-8') as f:\n"
        "    for row in debug_maps:\n"
        "        f.write(json.dumps(row, ensure_ascii=False) + '\\n')\n"
        "print('Saved', len(hyps_maps), 'MAPS predictions')\n"
    ),
    md(
        "## Part 4 — Retrieval-Augmented Generation for MT\n\n"
        "### 4.A Architecture\n\n"
        "![RAG architecture](diagrams/rag_architecture.png)\n\n"
        "**Indexing (offline).** We embed the source side of the WMT16 *training*\n"
        "set with a multilingual sentence encoder\n"
        "(`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim) and store the\n"
        "L2-normalized vectors in a FAISS `IndexFlatIP`. The metadata table maps\n"
        "each row to its `(src, tgt)` pair on disk. We cap the index at 200 K\n"
        "pairs after light length filtering to keep encoding under 5 minutes on\n"
        "CPU; the cap is set in `CONFIG['rag_max_index']`.\n\n"
        "**Retrieval (online).** At translation time we embed the test source\n"
        "with the same encoder and run an inner-product search. Inner product on\n"
        "L2-normalized vectors is exactly cosine similarity, so the search is a\n"
        "standard dense-retrieval setup.\n\n"
        "**Example selection.** Dense retrieval over MT corpora often returns\n"
        "near-paraphrases — five almost-identical sentences as five exemplars\n"
        "wastes the in-context budget. We over-retrieve `3·K` candidates and run\n"
        "a greedy MMR-lite filter that drops a candidate if its cosine similarity\n"
        "to any already-selected exemplar exceeds 0.92. The first 5 survivors\n"
        "form the demonstration set.\n\n"
        "**Generation.** The 5 retrieved `(src, tgt)` pairs are concatenated as\n"
        "few-shot exemplars in the chat prompt, the test source goes last, and\n"
        "the LLM produces the translation in one decode pass.\n"
    ),
    md(
        "### 4.B Build the FAISS index from the training side\n\n"
        "This cell is the slowest non-LLM step (~5 min on CPU for 200 K\n"
        "sentences); we run it once and the index persists in `outputs/`."
    ),
    code(
        "from src.data import filter_by_length\n"
        "from src.rag import RAGIndex, RAGIndexConfig\n"
        "\n"
        "train = filter_by_length(ds['train'])\n"
        "if len(train) > CONFIG['rag_max_index']:\n"
        "    train = train.shuffle(seed=42).select(range(CONFIG['rag_max_index']))\n"
        "\n"
        "pairs = [(r['src'], r['tgt']) for r in train]\n"
        "print(f'Indexing {len(pairs)} training pairs ...')\n"
        "\n"
        "rag = RAGIndex(RAGIndexConfig(\n"
        "    encoder_name=CONFIG['encoder_name'],\n"
        "    index_path=f\"{CONFIG['outputs_dir']}/rag_index.faiss\",\n"
        "    meta_path =f\"{CONFIG['outputs_dir']}/rag_meta.pkl\",\n"
        "))\n"
        "rag.build(pairs)\n"
        "len(pairs)\n"
    ),
    md(
        "### 4.B Inspect a single retrieval\n\n"
        "Pick one test sentence and look at the 5 examples MMR ends up selecting.\n"
        "If the top picks look thematically similar but lexically diverse,\n"
        "retrieval is doing what we want."
    ),
    code(
        "shots = rag.select_with_mmr(\n"
        "    items[0]['src'],\n"
        "    k=CONFIG['rag_k'],\n"
        "    mmr_threshold=CONFIG['rag_mmr_threshold'],\n"
        ")\n"
        "print(f'Query: {items[0][\"src\"]}\\n')\n"
        "for i, (s, t) in enumerate(shots, 1):\n"
        "    print(f'[{i}] {s}\\n    -> {t}\\n')\n"
    ),
    md(
        "### 4.B Full-test-set RAG generation\n\n"
        "Retrieve 5 shots per test sentence, then issue a single batched\n"
        "translation pass through Qwen 2.5."
    ),
    code(
        "from src.translator import rag_translate\n"
        "\n"
        "hyps_rag = []\n"
        "for s in tqdm(range(0, len(items), BATCH), desc='RAG'):\n"
        "    batch = items[s:s + BATCH]\n"
        "    retrieved = [\n"
        "        rag.select_with_mmr(\n"
        "            it['src'],\n"
        "            k=CONFIG['rag_k'],\n"
        "            mmr_threshold=CONFIG['rag_mmr_threshold'],\n"
        "        )\n"
        "        for it in batch\n"
        "    ]\n"
        "    hyps_rag.extend(r.hypothesis for r in rag_translate(backend, batch, retrieved, gen_cfg))\n"
        "save_predictions(f\"{CONFIG['outputs_dir']}/preds_rag.jsonl\",\n"
        "                 items, hyps_rag, extra={'system': 'rag_5shot'})\n"
        "print('Saved', len(hyps_rag), 'RAG predictions')\n"
    ),
    md(
        "## Part 5 — Experimental Comparison\n\n"
        "We score the three systems (zero-shot, MAPS, RAG-5-shot) with the\n"
        "reference-based **COMET-22** checkpoint (`Unbabel/wmt22-comet-da`), which\n"
        "is the same metric reported in the MAPS paper. Loading COMET takes ~30 s\n"
        "and the scoring run is ~10–15 minutes per system on RTX 6000 Ada\n"
        "for the full 3 000-segment test set.\n"
    ),
    code(
        "from src.evaluate import comet_score, load_predictions, save_scores\n"
        "import pandas as pd\n"
        "\n"
        "PRED_FILES = [\n"
        "    ('zero_shot', f\"{CONFIG['outputs_dir']}/preds_zeroshot.jsonl\"),\n"
        "    ('maps',      f\"{CONFIG['outputs_dir']}/preds_maps.jsonl\"),\n"
        "    ('rag_5shot', f\"{CONFIG['outputs_dir']}/preds_rag.jsonl\"),\n"
        "]\n"
        "\n"
        "summary = []\n"
        "for system, path in PRED_FILES:\n"
        "    rows = load_predictions(path)\n"
        "    res = comet_score(\n"
        "        [r['src'] for r in rows],\n"
        "        [r['hypothesis'] for r in rows],\n"
        "        [r['tgt'] for r in rows],\n"
        "        model_name=CONFIG['comet_model'],\n"
        "        batch_size=CONFIG['comet_batch_size'],\n"
        "        gpus=CONFIG['comet_gpus'],\n"
        "    )\n"
        "    save_scores(f\"{CONFIG['results_dir']}/comet_{system}.json\", system, res)\n"
        "    summary.append({'system': system, 'comet_22': res.system_score,\n"
        "                    'n': len(rows)})\n"
        "\n"
        "df = pd.DataFrame(summary)\n"
        "df.to_csv(f\"{CONFIG['results_dir']}/summary.csv\", index=False)\n"
        "df\n"
    ),
    md(
        "### Discussion (Part 5)\n\n"
        "Once the table above is populated, copy the three numbers into\n"
        "`report.pdf` and discuss:\n\n"
        "- **Translation improvements.** Compute `Δ COMET` for MAPS and RAG over\n"
        "  the zero-shot baseline. Comment on which approach helps more, and\n"
        "  whether RAG and MAPS are complementary (e.g. domain coverage vs.\n"
        "  reasoning).\n"
        "- **Computational overhead.** Zero-shot ≈ 1 LLM call/segment, MAPS ≈ 8\n"
        "  (3 mining + 4 candidate + 1 judge), RAG ≈ 1 + 1 retrieval. Latency\n"
        "  numbers from the run timers go here.\n"
        "- **Strengths and limitations of RAG-based prompting.** Strengths:\n"
        "  dynamic, domain-adaptive context with negligible LLM-side cost.\n"
        "  Limitations: only as good as the indexed corpus; risk of near-duplicate\n"
        "  retrieval (mitigated here by MMR); embedding model quality bottlenecks\n"
        "  cross-lingual retrieval.\n\n"
        "Plot of per-segment COMET histograms is generated below for the report."
    ),
    code(
        "import json\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "scores = {}\n"
        "for system, _ in PRED_FILES:\n"
        "    with open(f\"{CONFIG['results_dir']}/comet_{system}.json\", encoding='utf-8') as f:\n"
        "        scores[system] = json.load(f)['segment_scores']\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "for system, vals in scores.items():\n"
        "    ax.hist(vals, bins=30, alpha=0.45, label=f'{system} (mean={sum(vals)/len(vals):.3f})')\n"
        "ax.set_xlabel('Segment COMET-22 score')\n"
        "ax.set_ylabel('Frequency')\n"
        "ax.set_title('Per-segment COMET distribution — WMT16 EN→TR')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.savefig(f\"{CONFIG['results_dir']}/comet_histogram.png\", dpi=160)\n"
        "plt.show()\n"
    ),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("mt_prompting_rag.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Wrote {out} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
