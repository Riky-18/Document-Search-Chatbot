"""Tests for PDF reader and text extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pdf_reader import PdfReadError, ScannedPdfError, extract_pages


def _pdf_escape(text: str) -> str:
    """Escape special characters for standard PDF string literal."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_simple_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal valid PDF-1.4 file in memory. Empty strings create blank pages."""
    objects: list[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []

    for text in page_texts:
        if text:
            stream = f"BT /F1 16 Tf 72 700 Td ({_pdf_escape(text)}) Tj ET".encode("latin-1")
            content_id = add_obj(
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream
                + b"\nendstream"
            )
            page_body = (
                b"<< /Type /Page /Parent PAGES_ID 0 R /MediaBox [0 0 612 792] "
                + f"/Contents {content_id} 0 R ".encode("ascii")
                + f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode("ascii")
            )
        else:
            page_body = (
                b"<< /Type /Page /Parent PAGES_ID 0 R /MediaBox [0 0 612 792] "
                + f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode("ascii")
            )
        page_ids.append(add_obj(page_body))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii"))
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    patched = [body.replace(b"PAGES_ID", str(pages_id).encode("ascii")) for body in objects]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(patched, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("ascii")
        out += body
        out += b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(patched) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer << /Size {len(patched) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def write_pdf(path: Path, page_texts: list[str]) -> Path:
    """Helper to write a generated PDF to disk."""
    path.write_bytes(build_simple_pdf(page_texts))
    return path


def test_extracts_multipage_pdf(tmp_path: Path) -> None:
    """Test extracting text across multiple pages."""
    pdf = write_pdf(tmp_path / "multi.pdf", ["Page one text", "Page two text"])

    pages = extract_pages(pdf)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[0]["text"] == "Page one text"
    assert pages[1]["page_number"] == 2
    assert pages[1]["text"] == "Page two text"


def test_page_numbers_start_at_one(tmp_path: Path) -> None:
    """Test that page numbering starts at 1."""
    pdf = write_pdf(tmp_path / "numbered.pdf", ["Alpha", "Beta", "Gamma"])

    pages = extract_pages(pdf)

    assert [page["page_number"] for page in pages] == [1, 2, 3]


def test_empty_page_is_skipped(tmp_path: Path) -> None:
    """Test that blank/whitespace-only pages are skipped while preserving 1-based page numbers."""
    pdf = write_pdf(tmp_path / "gap.pdf", ["First page", "   \n\t  ", "Third page"])

    pages = extract_pages(pdf)

    assert len(pages) == 2
    assert pages[0] == {"page_number": 1, "text": "First page"}
    assert pages[1] == {"page_number": 3, "text": "Third page"}


def test_missing_file_raises_pdf_read_error(tmp_path: Path) -> None:
    """Test that a non-existent file raises PdfReadError."""
    missing_file = tmp_path / "non_existent.pdf"

    with pytest.raises(PdfReadError, match="PDF file not found"):
        extract_pages(missing_file)


def test_corrupt_file_raises_pdf_read_error(tmp_path: Path) -> None:
    """Test that a corrupt or invalid file raises PdfReadError."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"Not a real PDF stream or header \x00\xff\xfe")

    with pytest.raises(PdfReadError, match="Corrupt or unreadable PDF|Failed to read PDF"):
        extract_pages(corrupt_file)


def test_blank_pdf_raises_scanned_pdf_error(tmp_path: Path) -> None:
    """Test that a PDF with no extractable text raises ScannedPdfError."""
    pdf = write_pdf(tmp_path / "blank.pdf", ["", "   "])

    with pytest.raises(ScannedPdfError, match="No extractable text found"):
        extract_pages(pdf)


def test_normalizes_whitespace_and_null_bytes(tmp_path: Path) -> None:
    """Test that null bytes are stripped and whitespace is collapsed."""
    pdf = write_pdf(tmp_path / "normalized.pdf", ["Hello\x00   world \n\n from   PDF"])

    pages = extract_pages(pdf)

    assert len(pages) == 1
    assert pages[0]["text"] == "Hello world from PDF"
