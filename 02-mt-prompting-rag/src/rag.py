"""FAISS-backed dense retrieval for RAG-based few-shot translation.

Design (matches Part 4 of the project design):

* **Indexing** — we embed the **source side** of the WMT16 *training* set
  (after light length filtering) using a multilingual sentence encoder and
  store the L2-normalized vectors in a FAISS ``IndexFlatIP`` (cosine sim).
  Each vector keeps a pointer to the (src, tgt) pair on disk.

* **Retrieval** — at translation time, the same encoder embeds the test
  source sentence and we top-K it against the index.

* **Example selection** — to reduce near-duplicate exemplars (a common
  failure mode in dense retrieval over noisy MT corpora), we apply a simple
  MMR-style filter: we over-retrieve K' = 3K and greedily keep examples
  whose pairwise cosine similarity to already-selected ones is below
  ``mmr_threshold``.

* **Integration** — the resulting (src, tgt) pairs are passed to
  :func:`src.prompts.build_rag_few_shot` to form a 5-shot prompt.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class RAGIndexConfig:
    encoder_name: str = DEFAULT_ENCODER
    index_path: str = "outputs/rag_index.faiss"
    meta_path: str = "outputs/rag_meta.pkl"
    batch_size: int = 256
    normalize: bool = True


class RAGIndex:
    """Thin wrapper around a FAISS flat index + on-disk metadata."""

    def __init__(self, cfg: RAGIndexConfig | None = None):
        self.cfg = cfg or RAGIndexConfig()
        self._encoder = None
        self._index = None
        self._meta: List[Tuple[str, str]] = []

    # -- encoder ------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.cfg.encoder_name)
        return self._encoder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        enc = self._get_encoder()
        vecs = enc.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        return vecs.astype("float32")

    # -- build / load -------------------------------------------------------

    def build(self, pairs: Sequence[Tuple[str, str]]) -> None:
        """Index `pairs` of (src, tgt). The src side is what we search over."""
        import faiss

        srcs = [p[0] for p in pairs]
        vecs = self.encode(srcs)
        dim = vecs.shape[1]
        # Inner product on L2-normalized vectors == cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self._index = index
        self._meta = list(pairs)
        self._save()

    def _save(self) -> None:
        import faiss
        os.makedirs(os.path.dirname(self.cfg.index_path), exist_ok=True)
        faiss.write_index(self._index, self.cfg.index_path)
        with open(self.cfg.meta_path, "wb") as f:
            pickle.dump(self._meta, f)

    def load(self) -> None:
        import faiss
        self._index = faiss.read_index(self.cfg.index_path)
        with open(self.cfg.meta_path, "rb") as f:
            self._meta = pickle.load(f)

    # -- retrieval ----------------------------------------------------------

    def search(self, query: str, k: int) -> List[Tuple[float, Tuple[str, str]]]:
        if self._index is None:
            raise RuntimeError("Call build() or load() first.")
        q = self.encode([query])
        sims, idxs = self._index.search(q, k)
        return [
            (float(sims[0, j]), self._meta[int(idxs[0, j])])
            for j in range(idxs.shape[1])
            if int(idxs[0, j]) >= 0
        ]

    def batch_search(self, queries: Sequence[str], k: int) -> List[List[Tuple[float, Tuple[str, str]]]]:
        if self._index is None:
            raise RuntimeError("Call build() or load() first.")
        q = self.encode(list(queries))
        sims, idxs = self._index.search(q, k)
        out: List[List[Tuple[float, Tuple[str, str]]]] = []
        for i in range(idxs.shape[0]):
            row = []
            for j in range(idxs.shape[1]):
                ix = int(idxs[i, j])
                if ix < 0:
                    continue
                row.append((float(sims[i, j]), self._meta[ix]))
            out.append(row)
        return out

    # -- selection ----------------------------------------------------------

    def select_with_mmr(
        self,
        query: str,
        k: int = 5,
        over_retrieve: int = 3,
        mmr_threshold: float = 0.92,
    ) -> List[Tuple[str, str]]:
        """Greedy MMR-lite: drop candidates too similar to already-picked ones.

        This avoids degenerate cases where the top-5 are five paraphrases of
        the same sentence — a known failure mode on noisy MT corpora."""
        candidates = self.search(query, k * over_retrieve)
        if not candidates:
            return []
        enc = self._get_encoder()
        cand_vecs = enc.encode(
            [c[1][0] for c in candidates],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        picked: List[int] = []
        for i, (_, _) in enumerate(candidates):
            if len(picked) >= k:
                break
            ok = True
            for p in picked:
                sim = float(np.dot(cand_vecs[i], cand_vecs[p]))
                if sim > mmr_threshold:
                    ok = False
                    break
            if ok:
                picked.append(i)
        return [candidates[i][1] for i in picked]

    def batch_select_with_mmr(
        self,
        queries: Sequence[str],
        k: int = 5,
        over_retrieve: int = 3,
        mmr_threshold: float = 0.92,
    ) -> List[List[Tuple[str, str]]]:
        return [
            self.select_with_mmr(q, k=k, over_retrieve=over_retrieve, mmr_threshold=mmr_threshold)
            for q in queries
        ]
