"""Extract text from the source history PDF.

Uses pdfplumber as primary (better Turkish character handling) with pypdf fallback.
"""

from pathlib import Path
from typing import List

import pdfplumber
import pypdf


def extract_text_pdfplumber(pdf_path: str) -> List[str]:
    """Return one string per page."""
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def extract_text_pypdf(pdf_path: str) -> List[str]:
    reader = pypdf.PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def load_document(pdf_path: str, prefer: str = "pdfplumber") -> str:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    extractor = extract_text_pdfplumber if prefer == "pdfplumber" else extract_text_pypdf
    pages = extractor(pdf_path)
    joined = "\n\n".join(p.strip() for p in pages if p and p.strip())
    return joined
