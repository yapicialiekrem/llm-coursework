"""Evaluate zero-shot vs RAG accuracy on TurkishMMLU History subset.

Usage:
    python -m src.task2_rag.evaluate_accuracy --config configs/rag_config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.task1_lora_mt.inference import load_model

from .embedder import Embedder
from .rag_pipeline import (
    build_rag_prompt,
    build_zero_shot_prompt,
    generate_answer,
    parse_answer,
)
from .retriever import Retriever
from .vectorstore import ChromaStore


def load_history_subset(name: str, subject: str, split: str):
    """Load TurkishMMLU/History.

    TurkishMMLU exposes one config per subject (Biology, History, ...) and within
    each config one or more splits. We just load by config + split.
    """
    ds = load_dataset(name, subject)
    if split in ds:
        return ds[split]
    # fall back to first available split if requested one isn't present
    first = next(iter(ds.keys()))
    print(f"[warn] split '{split}' not found, using '{first}'")
    return ds[first]


def normalize_example(ex: dict) -> dict | None:
    """Best-effort coercion to {question, choices: List[str], answer: 'A'..'E'}."""
    q = ex.get("question") or ex.get("Question") or ex.get("soru")
    choices = ex.get("choices")
    if choices is None:
        # try A..E columns
        letters = ["A", "B", "C", "D", "E"]
        choices = [ex[l] for l in letters if l in ex and ex[l] is not None]
    answer = ex.get("answer") or ex.get("Answer") or ex.get("correct_answer")
    if isinstance(answer, int):
        answer = "ABCDE"[answer]
    if not q or not choices or not answer:
        return None
    return {"question": q, "choices": list(choices), "answer": str(answer).strip().upper()[:1]}


def run_eval(model, tokenizer, dataset, retriever: Retriever | None, mode: str, max_new_tokens: int) -> dict:
    correct = 0
    total = 0
    records = []
    for ex in tqdm(dataset, desc=f"eval/{mode}"):
        norm = normalize_example(ex)
        if norm is None:
            continue
        if mode == "rag":
            ctx = [doc for doc, _ in retriever.retrieve(norm["question"])]
            prompt = build_rag_prompt(tokenizer, norm["question"], norm["choices"], ctx)
        else:
            prompt = build_zero_shot_prompt(tokenizer, norm["question"], norm["choices"])
        raw = generate_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        pred = parse_answer(raw)
        is_correct = pred == norm["answer"]
        correct += int(is_correct)
        total += 1
        records.append({"q": norm["question"][:120], "pred": pred, "gold": norm["answer"], "correct": is_correct})
    accuracy = correct / total if total else 0.0
    return {"mode": mode, "accuracy": accuracy, "correct": correct, "total": total, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    db_cfg = cfg["vector_db"]
    emb_cfg = cfg["embedding"]
    gen_cfg = cfg["generation"]
    ds_cfg = cfg["dataset"]
    ret_cfg = cfg["retrieval"]
    eval_cfg = cfg["evaluation"]

    print("[load] dataset")
    dataset = load_history_subset(ds_cfg["name"], ds_cfg["subject"], ds_cfg["split"])

    print("[load] vector store + embedder")
    store = ChromaStore(db_cfg["persist_dir"], db_cfg["collection_name"], db_cfg["distance"])
    embedder = Embedder(emb_cfg["model_name"], device=emb_cfg["device"], normalize=emb_cfg["normalize"], batch_size=emb_cfg["batch_size"])
    retriever = Retriever(embedder, store, top_k=ret_cfg["top_k"])

    print("[load] LLM")
    adapter = gen_cfg["llm_path"] if Path(gen_cfg["llm_path"]).exists() else None
    model, tokenizer = load_model(gen_cfg["base_model"], adapter_path=adapter, load_in_4bit=gen_cfg["load_in_4bit"])

    results = {}
    all_records = {}
    for mode in eval_cfg["modes"]:
        r = run_eval(model, tokenizer, dataset, retriever if mode == "rag" else None, mode, gen_cfg["max_new_tokens"])
        results[mode] = {k: v for k, v in r.items() if k != "records"}
        all_records[mode] = r["records"]
        print(f"[{mode}] accuracy = {r['accuracy']:.4f}  ({r['correct']}/{r['total']})")

    Path(eval_cfg["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_cfg["output_file"], "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # also dump per-example records — useful for error analysis in the report
    records_path = Path(eval_cfg["output_file"]).with_name("rag_eval_records.json")
    with open(records_path, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"[done] aggregate -> {eval_cfg['output_file']}")
    print(f"[done] per-example records -> {records_path}")


if __name__ == "__main__":
    main()
