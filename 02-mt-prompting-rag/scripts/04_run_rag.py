"""Run RAG-based dynamic 5-shot translation on the WMT16 test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data import load_wmt16, sample_subset
from src.evaluate import save_predictions
from src.rag import RAGIndex, RAGIndexConfig
from src.translator import GenConfig, VLLMBackend, rag_translate
from src.utils import timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en-tr", choices=["en-tr", "tr-en"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mmr_threshold", type=float, default=0.92)
    ap.add_argument("--index_path", default="outputs/rag_index.faiss")
    ap.add_argument("--meta_path", default="outputs/rag_meta.pkl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", default="outputs/preds_rag.jsonl")
    args = ap.parse_args()

    print(f"[1/4] Loading WMT16 {args.direction} test split ...")
    ds = load_wmt16(args.direction)
    test = ds["test"]
    if args.limit:
        test = sample_subset(test, args.limit)
    items = [dict(r) for r in test]
    print(f"      -> {len(items)} test pairs")

    print(f"[2/4] Loading RAG index from {args.index_path} ...")
    rag = RAGIndex(RAGIndexConfig(index_path=args.index_path, meta_path=args.meta_path))
    rag.load()

    print(f"[3/4] Loading {args.model} via vLLM ...")
    backend = VLLMBackend(model_name=args.model)
    cfg = GenConfig(max_new_tokens=256, temperature=0.0)

    print(f"[4/4] Retrieving & translating ...")
    hyps = []
    with timer("RAG translation"):
        for start in tqdm(range(0, len(items), args.batch_size)):
            batch = items[start:start + args.batch_size]
            retrieved = [
                rag.select_with_mmr(it["src"], k=args.k, mmr_threshold=args.mmr_threshold)
                for it in batch
            ]
            results = rag_translate(backend, batch, retrieved, cfg)
            hyps.extend(r.hypothesis for r in results)

    save_predictions(args.output, items, hyps, extra={"system": "rag_5shot"})
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
