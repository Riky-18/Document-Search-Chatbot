"""Interactive CLI search script to index a PDF and query the vector store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure backend root is on sys.path when running as a direct script
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.chunker import chunk_pages
from app.pdf_reader import PdfReadError, ScannedPdfError, extract_pages
from app.vector_store import (
    add_chunks,
    get_stats,
    is_file_indexed,
    load,
    reset,
    save,
    search,
)


def _safe_print(text: str = "") -> None:
    """Print safely to standard output without crashing on Windows encoding limits."""
    try:
        print(text)
    except (UnicodeEncodeError, OSError):
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            safe_bytes = text.encode(encoding, errors="replace")
            sys.stdout.buffer.write(safe_bytes + b"\n")
            sys.stdout.flush()
        except Exception:
            print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def main() -> int:
    """Index or load a PDF and run an interactive query loop."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Index a PDF into the FAISS vector store and interactively search it."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the PDF file to index (defaults to test_pdfs/sample.pdf if present)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        dest="top_k",
        help="Number of nearest chunks to retrieve per search (default: 4)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-chunking and re-embedding even if the file is already indexed",
    )

    args = parser.parse_args()

    # Determine target PDF path
    raw_path = args.pdf_path
    if not raw_path:
        # Check standard locations
        candidates = [
            Path("../test_pdfs/sample.pdf"),
            Path("test_pdfs/sample.pdf"),
            Path("../../test_pdfs/sample.pdf"),
        ]
        for c in candidates:
            if c.exists():
                raw_path = str(c)
                break

    if not raw_path:
        _safe_print("No PDF path provided.")
        raw_path = input("Enter path to PDF file: ").strip().strip('"').strip("'")
        if not raw_path:
            _safe_print("No PDF path specified. Exiting.")
            return 1

    pdf_file = Path(raw_path).resolve()
    if not pdf_file.is_file():
        _safe_print(f"Error: File not found: {pdf_file}")
        return 1

    _safe_print(f"\n==================================================")
    _safe_print(f"Target Document: {pdf_file.name}")
    _safe_print(f"==================================================")

    # 1. On startup, load existing saved index from disk
    load()
    stats = get_stats()

    # 2. Check if already indexed
    already_indexed = is_file_indexed(pdf_file.name)

    if already_indexed and not args.reindex:
        _safe_print(
            f"Document '{pdf_file.name}' is already indexed in FAISS."
        )
        _safe_print(
            f"Skipping chunking and embedding -> going straight to search prompt."
        )
        _safe_print(
            f"Index stats: {stats['documents']} document(s), "
            f"{stats['pages']} page(s), {stats['chunks']} chunk(s) loaded."
        )
    else:
        if args.reindex:
            _safe_print("Force re-index requested (--reindex). Clearing previous index...")
            reset()

        _safe_print("Extracting pages from PDF...")
        try:
            pages = extract_pages(pdf_file)
        except (PdfReadError, ScannedPdfError) as exc:
            _safe_print(f"Error extracting pages: {exc}")
            return 1

        _safe_print(f"Extracted {len(pages)} page(s) with text.")
        chunks = chunk_pages(pages, file_name=pdf_file.name)
        _safe_print(f"Generated {len(chunks)} chunk(s).")

        if not chunks:
            _safe_print("No text chunks generated. Exiting.")
            return 1

        _safe_print("Embedding and adding chunks to FAISS vector store...")
        try:
            add_chunks(chunks)
            save()
        except Exception as exc:
            _safe_print(f"\nFailed to index chunks: {exc}")
            _safe_print("Please ensure your GEMINI_API_KEY is correctly set in backend/.env")
            return 1

        stats = get_stats()
        _safe_print(
            f"\nIndex stats: {stats['documents']} document(s), "
            f"{stats['pages']} page(s), {stats['chunks']} chunk(s) stored."
        )

    # 3. Interactive Query Loop
    _safe_print("\n==================================================")
    _safe_print("Ready for search queries. Type 'quit' or 'exit' to stop.")
    _safe_print("==================================================")

    while True:
        try:
            query = input("\nSearch question: ").strip()
        except (KeyboardInterrupt, EOFError):
            _safe_print("\nExiting search.")
            break

        if not query:
            continue

        if query.lower() in ["quit", "exit", "q"]:
            _safe_print("Exiting search. Goodbye!")
            break

        try:
            results = search(query, k=args.top_k)
        except Exception as exc:
            _safe_print(f"Search error: {exc}")
            continue

        if not results:
            _safe_print("No relevant chunks found.")
            continue

        _safe_print(f"\nFound {len(results)} matching chunk(s):")
        for rank, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            file_name = meta.get("file_name", "unknown")
            page_num = meta.get("page_number", "?")
            chunk_idx = meta.get("chunk_index", "?")
            score = res.get("score", 0.0)
            text = res.get("text", "")

            _safe_print(f"\n[{rank}] Score: {score:.4f} | File: {file_name} | Page: {page_num} | Chunk: {chunk_idx}")
            _safe_print(f"    {text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
