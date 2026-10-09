"""Chunk extracted document pages into smaller text segments using LangChain."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.pdf_reader import PdfReadError, ScannedPdfError, extract_pages

logger = logging.getLogger(__name__)


def chunk_pages(
    pages: list[dict],
    file_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split pages into smaller text chunks without crossing page boundaries.

    Args:
        pages: List of dictionaries with 'page_number' and 'text'.
        file_name: Originating file name to attach to chunk metadata.
        chunk_size: Maximum characters per chunk (default: 800).
        chunk_overlap: Overlapping character count between consecutive chunks (default: 100).

    Returns:
        List of chunks with keys:
            - text: str
            - file_name: str
            - page_number: int
            - chunk_index: int
    """
    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        page_number = page.get("page_number", 0)
        page_text = page.get("text", "")
        if not page_text or not page_text.strip():
            continue

        raw_splits = splitter.split_text(page_text)
        for split in raw_splits:
            cleaned = split.strip()
            if not cleaned:
                continue
            chunks.append({
                "text": cleaned,
                "file_name": file_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    logger.info(
        "Generated %d chunk(s) across %d page(s) for '%s'",
        len(chunks),
        len(pages),
        file_name,
    )
    return chunks


def _safe_print(text: str = "") -> None:
    """Print text safely, ensuring non-ASCII or unsupported characters never crash stdout."""
    try:
        print(text)
    except (UnicodeEncodeError, OSError):
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            safe_bytes = text.encode(encoding, errors="replace")
            sys.stdout.buffer.write(safe_bytes + b"\n")
            sys.stdout.flush()
        except Exception:
            # Fallback to ASCII representation
            print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def main(argv: list[str] | None = None) -> int:
    """CLI to read and chunk a PDF document."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Extract and chunk text from a PDF.")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        dest="chunk_size",
        help=f"Chunk size (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        dest="chunk_overlap",
        help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})",
    )

    args = parser.parse_args(argv)
    pdf_path = Path(args.pdf_path)

    try:
        pages = extract_pages(pdf_path)
    except (PdfReadError, ScannedPdfError) as exc:
        _safe_print(f"Error reading PDF: {exc}")
        return 1

    chunks = chunk_pages(
        pages=pages,
        file_name=pdf_path.name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    total_chunks = len(chunks)
    if total_chunks == 0:
        _safe_print(f"File: {pdf_path.name}")
        _safe_print("Total chunks: 0")
        _safe_print("Average chunk length: 0.0 characters")
        return 0

    avg_len = sum(len(c["text"]) for c in chunks) / total_chunks
    _safe_print(f"File: {pdf_path.name}")
    _safe_print(f"Total chunks: {total_chunks}")
    _safe_print(f"Average chunk length: {avg_len:.1f} characters")

    sample_count = min(3, total_chunks)
    _safe_print(f"\nShowing {sample_count} sample chunk(s):")
    for i in range(sample_count):
        chunk = chunks[i]
        text_preview = chunk["text"]
        _safe_print(f"\n--- Chunk {chunk['chunk_index']} (Page {chunk['page_number']}, {len(text_preview)} chars) ---")
        _safe_print(text_preview)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
