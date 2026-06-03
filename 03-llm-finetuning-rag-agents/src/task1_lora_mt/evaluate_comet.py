"""COMET evaluation for EN<->TR translation.

Compares: (a) base Qwen2.5-7B-Instruct zero-shot vs (b) LoRA fine-tuned.

Usage:
    python -m src.task1_lora_mt.evaluate_comet --config configs/lora_config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from comet import download_model, load_from_checkpoint

from .data_loader import load_wmt16_tr_en
from .inference import load_model, translate_batch


def evaluate(model, tokenizer, test_ds, direction: str, gen_cfg: dict, batch_size: int):
    if direction == "en2tr":
        sources = [ex["translation"]["en"] for ex in test_ds]
        refs = [ex["translation"]["tr"] for ex in test_ds]
    else:
        sources = [ex["translation"]["tr"] for ex in test_ds]
        refs = [ex["translation"]["en"] for ex in test_ds]
    hyps = translate_batch(
        model,
        tokenizer,
        sources,
        direction=direction,
        max_new_tokens=gen_cfg["max_new_tokens"],
        batch_size=batch_size,
    )
    return sources, refs, hyps


def comet_score(comet_model, sources, refs, hyps, batch_size: int = 16) -> float:
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    result = comet_model.predict(data, batch_size=batch_size, gpus=1)
    return float(result.system_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Override adapter path; default is training output_dir.")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only evaluate the base model (no LoRA).")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the base model evaluation (assumes baseline JSON already exists).")
    parser.add_argument("--output", type=str, default="outputs/logs/comet_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg["evaluation"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    print(f"[comet] downloading {eval_cfg['comet_model']}")
    ckpt = download_model(eval_cfg["comet_model"])
    comet_model = load_from_checkpoint(ckpt)

    _, _, test_raw = load_wmt16_tr_en(
        train_samples=1, eval_samples=1,
        test_samples=data_cfg["test_samples"], seed=42,
    )

    results: dict = {}

    # ---- Baseline (zero-shot) ----
    if not args.skip_baseline:
        print("[eval] loading base model")
        model, tok = load_model(model_cfg["name"], adapter_path=None, load_in_4bit=True)
        for direction in ["en2tr", "tr2en"]:
            print(f"[eval-base] {direction}")
            src, ref, hyp = evaluate(model, tok, test_raw, direction, eval_cfg["generation"], eval_cfg["batch_size"])
            score = comet_score(comet_model, src, ref, hyp, eval_cfg["batch_size"])
            results[f"base_{direction}_comet"] = score
            print(f"  COMET = {score:.4f}")
        del model, tok
    else:
        print("[eval] skipping baseline (reuse existing JSON if any)")
        # Try to merge prior baseline scores if a comet_baseline.json sits next to our output
        prior = Path(args.output).parent / "comet_baseline.json"
        if prior.exists():
            with open(prior) as f:
                results.update(json.load(f))
            print(f"[eval] loaded prior baseline from {prior}: {results}")

    # ---- LoRA fine-tuned ----
    if not args.baseline_only:
        adapter = args.adapter or cfg["training"]["output_dir"]
        print(f"[eval] loading fine-tuned model with adapter {adapter}")
        model, tok = load_model(model_cfg["name"], adapter_path=adapter, load_in_4bit=True)
        for direction in ["en2tr", "tr2en"]:
            print(f"[eval-lora] {direction}")
            src, ref, hyp = evaluate(model, tok, test_raw, direction, eval_cfg["generation"], eval_cfg["batch_size"])
            score = comet_score(comet_model, src, ref, hyp, eval_cfg["batch_size"])
            results[f"lora_{direction}_comet"] = score
            print(f"  COMET = {score:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results -> {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
