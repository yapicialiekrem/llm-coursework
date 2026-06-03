"""Thin retriever wrapping the embedder + Chroma store."""

from typing import List, Tuple

from .embedder import Embedder
from .vectorstore import ChromaStore


class Retriever:
    def __init__(self, embedder: Embedder, store: ChromaStore, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        k = top_k or self.top_k
        q_emb = self.embedder.embed_query(query)
        res = self.store.query(q_emb, top_k=k)
        docs = res["documents"][0]
        dists = res["distances"][0]
        return list(zip(docs, dists))
