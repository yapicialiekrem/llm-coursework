"""ChromaDB persistent vector store helper."""

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str, distance: str = "cosine"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )

    def add(self, ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[dict] | None = None):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(ids),
        )

    def query(self, query_embedding: List[float], top_k: int = 5) -> dict:
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def count(self) -> int:
        return self.collection.count()
