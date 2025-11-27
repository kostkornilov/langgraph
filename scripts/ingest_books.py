"""Ingest PDFs from `books/` and write chunked text to `data/chunks.jsonl`.

Upgrades:
- Uses a deterministic overlapping character splitter
    (defaults: 4800 characters, 600 overlap) tuned for Russian prose.
- Persists precise page ranges for every chunk by tracking cumulative
    character offsets across the PDF.
- Outputs rich metadata: chunk/page indices plus char spans to help
    debugging retrieval mismatches.

Run (Windows PowerShell):
    .\myenv\Scripts\Activate.ps1
    python scripts\ingest_books.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "books"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chunks.jsonl"

CHUNK_SIZE = int(os.environ.get("INGEST_CHUNK_CHARS", "4800"))
CHUNK_OVERLAP = int(os.environ.get("INGEST_CHUNK_OVERLAP", "600"))


def extract_text_from_pdf(path: Path) -> list[dict]:
    """Extract text from a PDF file using PyPDF2 and return a list of page dicts.

    Returns a list like: [{"page_index": 0, "text": "..."}, ...]
    """
    try:
        import PyPDF2
    except Exception as e:
        raise ImportError(
            "PyPDF2 is required to extract text from PDFs. Install it with: pip install PyPDF2"
        ) from e

    pages: list[dict] = []
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for i, page in enumerate(reader.pages):
            # page.extract_text() may return None on some pages; coerce to empty string
            text = page.extract_text() or ""
            pages.append({"page_index": i, "text": text})

    return pages


def _iter_text_slices(text: str):
    n = len(text)
    start = 0
    while start < n:
        max_end = min(start + CHUNK_SIZE, n)
        end = max_end
        preferred_start = start + CHUNK_SIZE // 2
        for sep in ["\n\n", "\n", " "]:
            idx = text.rfind(sep, preferred_start, max_end)
            if idx != -1 and idx > start:
                end = idx
                break
        if end <= start:
            end = max_end
        chunk_text = text[start:end].strip()
        if chunk_text:
            yield start, end, chunk_text
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, 0)
        if start == end:
            start = end


def build_chunks_from_pages(pages: List[dict], source_name: str) -> Iterable[dict]:
    """Chunk the PDF with overlap while keeping track of page spans."""

    sep = "\n\n"
    normalized_pages = []
    for page in pages:
        text = (page["text"] or "").replace("\r\n", "\n")
        normalized_pages.append({"index": page["page_index"], "text": text})

    parts: List[str] = []
    offsets: List[dict] = []
    cursor = 0
    for i, page in enumerate(normalized_pages):
        parts.append(page["text"])
        offsets.append({
            "page_index": page["index"],
            "start": cursor,
            "end": cursor + len(page["text"]),
        })
        cursor += len(page["text"])
        if i != len(normalized_pages) - 1:
            parts.append(sep)
            cursor += len(sep)

    full_text = "".join(parts)
    if not full_text.strip():
        return

    page_bounds = [off["end"] for off in offsets]

    def idx_to_page(char_idx: int) -> int:
        pos = bisect_right(page_bounds, char_idx)
        pos = min(max(pos, 0), len(offsets) - 1)
        return offsets[pos]["page_index"]

    for start_index, end_index, chunk_text in _iter_text_slices(full_text):
        yield {
            "page_index_start": idx_to_page(start_index),
            "page_index_end": idx_to_page(max(end_index - 1, start_index)),
            "text": chunk_text,
            "start_char": start_index,
            "end_char": end_index,
        }


def main() -> None:
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {BOOKS_DIR}. Place PDFs there and rerun.")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in {BOOKS_DIR}")

    out_f = OUT_FILE.open("w", encoding="utf-8")
    total_chunks = 0

    for pdf in pdfs:
        print(f"Processing {pdf.name}...")
        try:
            pages = extract_text_from_pdf(pdf)
        except ImportError as e:
            print(e)
            print("Aborting ingestion. Install PyPDF2 in your environment and retry.")
            sys.exit(2)

    for i, chunk in enumerate(build_chunks_from_pages(pages, pdf.name)):
            doc = {
                "id": str(uuid.uuid4()),
                "source": pdf.name,
                "chunk_index": i,
                "page_index_start": int(chunk.get("page_index_start", 0)),
                "page_index_end": int(chunk.get("page_index_end", chunk.get("page_index_start", 0))),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "text": chunk["text"].strip(),
            }
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_chunks += 1

    out_f.close()
    print(f"Wrote {total_chunks} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
