"""Small utilities shared by the scripts."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    print(f"[{label}] {time.time() - t0:.1f}s")


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
