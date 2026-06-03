"""Score one or more predictions JSONL files with COMET-22.

Usage:
    python scripts/05_evaluate.py \
        --pred outputs/preds_zeroshot.jsonl \
        --pred outputs/preds_maps.jsonl \
        --pred outputs/preds_rag.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import comet_score, load_predictions, save_scores
from src.utils import timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True,
                    help="Path(s) to predictions JSONL files. Repeatable.")
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    summary = []
    for pred_path in args.pred:
        rows = load_predictions(pred_path)
        if not rows:
            print(f"[skip] {pred_path} is empty")
            continue
        srcs = [r["src"] for r in rows]
        hyps = [r["hypothesis"] for r in rows]
        refs = [r["tgt"] for r in rows]
        system = rows[0].get("system", Path(pred_path).stem)

        print(f"[{system}] scoring {len(rows)} segments ...")
        with timer(f"COMET[{system}]"):
            result = comet_score(srcs, hyps, refs, model_name=args.model,
                                 batch_size=args.batch_size, gpus=args.gpus)
        print(f"   system COMET = {result.system_score:.4f}")

        out_path = Path(args.out_dir) / f"comet_{system}.json"
        save_scores(str(out_path), system, result)
        summary.append({"system": system, "comet": result.system_score,
                        "n": len(rows), "path": pred_path})

    Path(args.out_dir).mkdir(exist_ok=True)
    with open(Path(args.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for s in summary:
        print(f"  {s['system']:15s}  COMET={s['comet']:.4f}  n={s['n']}")


if __name__ == "__main__":
    main()
