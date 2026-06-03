"""Sentence-transformer embedding wrapper (default: BAAI/bge-m3)."""

from typing import List

import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda", normalize: bool = True, batch_size: int = 32):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return emb.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]
