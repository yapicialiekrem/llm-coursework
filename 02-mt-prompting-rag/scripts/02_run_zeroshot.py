"""Run zero-shot translation on the WMT16 test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.data import load_wmt16, sample_subset
from src.evaluate import save_predictions
from src.translator import GenConfig, VLLMBackend, zero_shot_translate
from src.utils import timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en-tr", choices=["en-tr", "tr-en"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full test set; otherwise take a deterministic subset.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", default="outputs/preds_zeroshot.jsonl")
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

    print(f"[3/3] Translating ...")
    hyps = []
    with timer("zero-shot translation"):
        for start in tqdm(range(0, len(items), args.batch_size)):
            batch = items[start:start + args.batch_size]
            results = zero_shot_translate(backend, batch, cfg)
            hyps.extend(r.hypothesis for r in results)

    save_predictions(args.output, items, hyps, extra={"system": "zero_shot"})
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
