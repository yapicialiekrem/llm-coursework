"""End-to-end indexer: PDF -> chunks -> embeddings -> ChromaDB.

Usage:
    python -m src.task2_rag.build_index --config configs/rag_config.yaml
"""

import argparse
import uuid
from pathlib import Path

import yaml

from .chunker import chunk_text
from .document_loader import load_document
from .embedder import Embedder
from .vectorstore import ChromaStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    kb = cfg["knowledge_base"]
    ck = cfg["chunking"]
    emb_cfg = cfg["embedding"]
    db_cfg = cfg["vector_db"]

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    for entry in kb["source_pdfs"]:
        pdf_path = entry["path"]
        source_id = entry["source_id"]
        print(f"[load] {pdf_path}")
        text = load_document(pdf_path)
        print(f"[load] {len(text):,} characters from {source_id}")

        print(f"[chunk] splitting {source_id}")
        chunks = chunk_text(
            text,
            chunk_size=ck["chunk_size"],
            chunk_overlap=ck["chunk_overlap"],
            separators=ck["separators"],
            tokenizer_name=emb_cfg["model_name"],
        )
        print(f"[chunk] {source_id}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
        all_metadatas.extend({"source": source_id, "chunk_idx": i} for i in range(len(chunks)))

    print(f"[chunk] total {len(all_chunks)} chunks across {len(kb['source_pdfs'])} books")

    print(f"[embed] {emb_cfg['model_name']}")
    embedder = Embedder(
        model_name=emb_cfg["model_name"],
        device=emb_cfg["device"],
        normalize=emb_cfg["normalize"],
        batch_size=emb_cfg["batch_size"],
    )
    vectors = embedder.embed(all_chunks)

    print(f"[store] {db_cfg['persist_dir']} / {db_cfg['collection_name']}")
    store = ChromaStore(
        persist_dir=db_cfg["persist_dir"],
        collection_name=db_cfg["collection_name"],
        distance=db_cfg["distance"],
    )
    ids = [str(uuid.uuid4()) for _ in all_chunks]
    store.add(ids=ids, texts=all_chunks, embeddings=vectors, metadatas=all_metadatas)
    print(f"[store] count = {store.count()}")


if __name__ == "__main__":
    main()
