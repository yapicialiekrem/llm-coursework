"""WMT16 English-Turkish dataset loading and preprocessing.

The WMT16 EN<->TR pair is exposed by HuggingFace under the `wmt16` builder with
the `tr-en` config. It contains the IWSLT-style news commentary + SETIMES2
parallel corpora used in the WMT 2016 news translation shared task.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset


WMT16_CONFIG = "tr-en"

# Light, reversible normalization. We deliberately avoid aggressive cleanup so
# that COMET scores reflect translation quality rather than tokenization.
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass
class Pair:
    src: str
    tgt: str
    src_lang: str
    tgt_lang: str

    def as_dict(self) -> dict:
        return {
            "src": self.src,
            "tgt": self.tgt,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
        }


def load_wmt16(direction: str = "en-tr") -> DatasetDict:
    """Load the WMT16 tr-en split as a DatasetDict with 'src'/'tgt' columns.

    direction: "en-tr" or "tr-en". Determines which side is source/target.
    """
    if direction not in {"en-tr", "tr-en"}:
        raise ValueError(f"Unsupported direction: {direction}")

    raw = load_dataset("wmt16", WMT16_CONFIG)

    src_lang, tgt_lang = direction.split("-")

    def project(example):
        pair = example["translation"]
        return {
            "src": _normalize(pair[src_lang]),
            "tgt": _normalize(pair[tgt_lang]),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }

    projected = DatasetDict()
    for split, ds in raw.items():
        projected[split] = ds.map(project, remove_columns=ds.column_names)
    return projected


def filter_by_length(
    ds: Dataset, min_chars: int = 10, max_chars: int = 400
) -> Dataset:
    """Drop empty / pathologically long sentences. Applied to train only."""
    def keep(example):
        s, t = example["src"], example["tgt"]
        if not s or not t:
            return False
        if len(s) < min_chars or len(t) < min_chars:
            return False
        if len(s) > max_chars or len(t) > max_chars:
            return False
        return True
    return ds.filter(keep)


def sample_subset(ds: Dataset, n: int, seed: int = 42) -> Dataset:
    """Deterministic random subset (used only for ablation; the H100 run uses
    the entire test set)."""
    if n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def dataset_stats(ds: DatasetDict) -> dict:
    out = {}
    for split, d in ds.items():
        src_lens = [len(r["src"].split()) for r in d]
        tgt_lens = [len(r["tgt"].split()) for r in d]
        out[split] = {
            "n_pairs": len(d),
            "src_tokens_mean": sum(src_lens) / max(len(src_lens), 1),
            "tgt_tokens_mean": sum(tgt_lens) / max(len(tgt_lens), 1),
            "src_tokens_max": max(src_lens, default=0),
            "tgt_tokens_max": max(tgt_lens, default=0),
        }
    return out


def iter_pairs(ds: Dataset) -> Iterable[Pair]:
    for row in ds:
        yield Pair(
            src=row["src"],
            tgt=row["tgt"],
            src_lang=row["src_lang"],
            tgt_lang=row["tgt_lang"],
        )
