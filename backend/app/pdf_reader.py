"""Extract text from PDF documents page by page."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError as PyPdfReadError

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


class PdfReadError(Exception):
    """Raised when a PDF file is missing, unreadable, or corrupt."""


class ScannedPdfError(Exception):
    """Raised when a PDF has no extractable text (e.g. scanned or image-only)."""


def _normalize_text(text: str) -> str:
    """Normalize whitespace and strip null bytes."""
    cleaned = text.replace("\x00", "")
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Extract non-empty text pages from a PDF.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        List of dicts: [{"page_number": int, "text": str}] with 1-based page numbers.

    Raises:
        PdfReadError: If the file does not exist, cannot be opened, or is corrupt.
        ScannedPdfError: If no extractable text is found across the entire PDF.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise PdfReadError(f"PDF file not found: {path}")

    logger.info("Extracting pages from: %s", path)

    try:
        reader = PdfReader(str(path))
        if len(reader.pages) == 0:
            raise ScannedPdfError(f"No pages found in PDF '{path}'.")

        pages: list[dict] = []
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            text = _normalize_text(raw_text)
            if not text:
                logger.debug("Skipping empty page %s in %s", page_number, path.name)
                continue
            pages.append({"page_number": page_number, "text": text})

    except (PdfReadError, ScannedPdfError):
        raise
    except (PyPdfReadError, OSError, ValueError) as exc:
        raise PdfReadError(f"Failed to read PDF '{path}': {exc}") from exc
    except Exception as exc:
        raise PdfReadError(f"Corrupt or unreadable PDF '{path}': {exc}") from exc

    if not pages:
        raise ScannedPdfError(
            f"No extractable text found in '{path}'. This appears to be a scanned or image-only PDF."
        )

    logger.info("Successfully extracted %s text page(s) from %s", len(pages), path.name)
    return pages


def main(argv: list[str] | None = None) -> int:
    """CLI interface to preview text extracted from a PDF."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python -m app.pdf_reader <file.pdf>", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        pages = extract_pages(args[0])
    except (PdfReadError, ScannedPdfError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Total pages extracted: {len(pages)}")
    for page in pages:
        preview = page["text"][:200]
        print(f"\n--- Page {page['page_number']} ---")
        print(preview)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
