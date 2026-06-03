"""Recursive chunking using LangChain's RecursiveCharacterTextSplitter.

Token-aware sizing via the HF tokenizer so chunk_size really means ~tokens.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


def build_splitter(chunk_size: int, chunk_overlap: int, separators: List[str], tokenizer_name: str = "BAAI/bge-m3"):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def token_len(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=token_len,
        keep_separator=False,
    )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, separators: List[str], tokenizer_name: str = "BAAI/bge-m3") -> List[str]:
    splitter = build_splitter(chunk_size, chunk_overlap, separators, tokenizer_name)
    return splitter.split_text(text)
