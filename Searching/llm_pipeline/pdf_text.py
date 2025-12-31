from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from llm_pipeline.utils import sanitize_surrogates


def _safe_page_text(page) -> str:
    try:
        t = page.extract_text() or ""
        return sanitize_surrogates(t)
    except Exception:
        return ""


def extract_pdf_text(
    pdf_path: str | Path,
    *,
    head_pages: int = 2,
    tail_pages: int = 2,
    max_chars: int = 60000,
) -> str:
    """
    Extract a compact, LLM-friendly PDF_TEXT:
    - first head_pages (title/authors/abstract usually here)
    - last tail_pages (references often here)
    - truncate to max_chars
    """
    pdf_path = Path(pdf_path)
    # pypdf may emit noisy warnings for slightly-malformed PDFs (common in the wild).
    # We prefer robust extraction over strict PDF conformance.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(pdf_path), strict=False)
    n = len(reader.pages)
    head_pages = max(0, min(head_pages, n))
    tail_pages = max(0, min(tail_pages, n - head_pages))

    head = []
    for i in range(head_pages):
        head.append(_safe_page_text(reader.pages[i]))

    tail = []
    for i in range(n - tail_pages, n):
        if 0 <= i < n and i >= head_pages:
            tail.append(_safe_page_text(reader.pages[i]))

    text = (
        f"[PDF_FILE] {pdf_path.name}\n"
        f"[PAGES] {n}\n\n"
        "===== BEGIN_HEAD =====\n"
        + "\n\n".join(head).strip()
        + "\n===== END_HEAD =====\n\n"
        "===== BEGIN_TAIL =====\n"
        + "\n\n".join(tail).strip()
        + "\n===== END_TAIL =====\n"
    )

    if len(text) > max_chars:
        text = text[: max_chars - 50] + "\n\n[TRUNCATED]\n"
    return text


