"""COMET evaluation utilities.

We use the reference-based **wmt22-comet-da** model (the standard COMET-22
checkpoint), which is what the original MAPS paper reports. It returns a
sentence-level score in roughly [0, 1] and a system-level mean.

The companion reference-free model (``wmt22-cometkiwi-da``) can be loaded by
passing ``model_name=...`` if you want to mirror MAPS's QE-based selection.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence


COMET_MODEL_DEFAULT = "Unbabel/wmt22-comet-da"


@dataclass
class CometResult:
    system_score: float
    segment_scores: List[float]


def comet_score(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str],
    model_name: str = COMET_MODEL_DEFAULT,
    batch_size: int = 16,
    gpus: int = 1,
) -> CometResult:
    """Run COMET evaluation and return system + per-segment scores."""
    from comet import download_model, load_from_checkpoint

    if not (len(sources) == len(hypotheses) == len(references)):
        raise ValueError("sources / hypotheses / references must align")

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return CometResult(
        system_score=float(out["system_score"]),
        segment_scores=[float(x) for x in out["scores"]],
    )


def save_predictions(
    path: str,
    items: Sequence[dict],
    hypotheses: Sequence[str],
    extra: dict | None = None,
) -> None:
    """Persist hypotheses + sources + references to a JSONL file. This is the
    artifact we feed into ``comet_score`` after a translation run."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = extra or {}
    with open(path, "w", encoding="utf-8") as f:
        for it, h in zip(items, hypotheses):
            rec = {
                "src": it["src"],
                "tgt": it["tgt"],
                "src_lang": it["src_lang"],
                "tgt_lang": it["tgt_lang"],
                "hypothesis": h,
            }
            rec.update(extra)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_predictions(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_scores(path: str, system: str, result: CometResult) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "system": system,
        "comet_system_score": result.system_score,
        "n_segments": len(result.segment_scores),
        "segment_scores": result.segment_scores,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
