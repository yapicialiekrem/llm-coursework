"""Run the MAPS pipeline (Multi-Aspect Prompting and Selection) on WMT16 test.

This script does 3 batched knowledge-mining passes + 4 batched candidate
generations + 1 batched judge pass per shard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data import load_wmt16, sample_subset
from src.evaluate import save_predictions
from src.translator import GenConfig, VLLMBackend, maps_translate
from src.utils import timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en-tr", choices=["en-tr", "tr-en"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", default="outputs/preds_maps.jsonl")
    ap.add_argument("--debug_output", default="outputs/preds_maps_debug.jsonl")
    args = ap.parse_args()

    print(f"[1/3] Loading WMT16 {args.direction} test split ...")
    ds = load_wmt16(args.direction)
    test = ds["test"]
    if args.limit:
        test = sample_subset(test, args.limit)
    items = [dict(r) for r in test]
    print(f"      -> {len(items)} test pairs")

    print(f"[2/3] Loading {args.model} via vLLM ...")
    backend = VLLMBackend(model_name=args.model)
    cfg = GenConfig(max_new_tokens=256, temperature=0.0)
    judge_cfg = GenConfig(max_new_tokens=4, temperature=0.0)

    print(f"[3/3] Running MAPS pipeline ...")
    hyps = []
    debug_rows = []
    with timer("MAPS translation"):
        for start in tqdm(range(0, len(items), args.batch_size)):
            batch = items[start:start + args.batch_size]
            results = maps_translate(backend, batch, cfg, judge_cfg)
            for it, r in zip(batch, results):
                hyps.append(r.hypothesis)
                debug_rows.append({
                    "src": it["src"],
                    "tgt": it["tgt"],
                    "hypothesis": r.hypothesis,
                    **r.intermediate,
                })

    save_predictions(args.output, items, hyps, extra={"system": "maps"})
    Path(args.debug_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.debug_output, "w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved hypotheses -> {args.output}")
    print(f"Saved debug      -> {args.debug_output}")


if __name__ == "__main__":
    main()
