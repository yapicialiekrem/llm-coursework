"""Build the FAISS index from the WMT16 training set.

Run once before any RAG translation. Persists `outputs/rag_index.faiss` and
`outputs/rag_meta.pkl` that downstream scripts load.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import filter_by_length, load_wmt16
from src.rag import RAGIndex, RAGIndexConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en-tr", choices=["en-tr", "tr-en"])
    ap.add_argument("--max_pairs", type=int, default=200_000,
                    help="Cap the indexed corpus for memory / speed.")
    ap.add_argument("--index_path", default="outputs/rag_index.faiss")
    ap.add_argument("--meta_path", default="outputs/rag_meta.pkl")
    ap.add_argument("--encoder", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    print(f"[1/3] Loading WMT16 {args.direction} ...")
    ds = load_wmt16(args.direction)
    train = filter_by_length(ds["train"])
    if args.max_pairs and len(train) > args.max_pairs:
        train = train.shuffle(seed=42).select(range(args.max_pairs))

    pairs = [(r["src"], r["tgt"]) for r in train]
    print(f"      -> {len(pairs)} (src, tgt) pairs")

    cfg = RAGIndexConfig(
        encoder_name=args.encoder,
        index_path=args.index_path,
        meta_path=args.meta_path,
    )
    print(f"[2/3] Encoding with {args.encoder} ...")
    idx = RAGIndex(cfg)
    idx.build(pairs)
    print(f"[3/3] Saved index -> {args.index_path}")
    print(f"               meta -> {args.meta_path}")


if __name__ == "__main__":
    main()
