"""Tests for document page chunker."""

from __future__ import annotations

import pytest

from app.chunker import chunk_pages


def test_chunks_never_cross_pages() -> None:
    """Verify that chunks generated from separate pages never mix text across page boundaries."""
    # Create two pages with repetitive distinct markers long enough to require multiple chunks
    page_1_marker = "ALPHA " * 150
    page_2_marker = "BETA " * 150
    pages = [
        {"page_number": 1, "text": page_1_marker},
        {"page_number": 2, "text": page_2_marker},
    ]

    chunks = chunk_pages(pages, file_name="sample.pdf", chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 2
    for chunk in chunks:
        if chunk["page_number"] == 1:
            assert "BETA" not in chunk["text"]
            assert "ALPHA" in chunk["text"]
        elif chunk["page_number"] == 2:
            assert "ALPHA" not in chunk["text"]
            assert "BETA" in chunk["text"]


def test_metadata_is_correct() -> None:
    """Verify that all metadata fields (file_name, page_number, chunk_index) are correctly assigned."""
    pages = [
        {"page_number": 1, "text": "Page one short text."},
        {"page_number": 2, "text": "Page two text that will be chunked into multiple pieces. " * 15},
    ]
    file_name = "test_document.pdf"

    chunks = chunk_pages(pages, file_name=file_name, chunk_size=150, chunk_overlap=20)

    assert len(chunks) >= 3
    # chunk_index must start at 0 and increase in strict sequential order
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))

    for chunk in chunks:
        assert chunk["file_name"] == file_name
        assert isinstance(chunk["page_number"], int)
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0


def test_overlap_works() -> None:
    """Verify that consecutive chunks from the same page share overlapping text."""
    # A single continuous text to split
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    pages = [{"page_number": 1, "text": text}]

    chunk_size = 120
    chunk_overlap = 30
    chunks = chunk_pages(pages, file_name="overlap_test.pdf", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    # Check that adjacent chunks share text at boundary
    for i in range(len(chunks) - 1):
        c1_text = chunks[i]["text"]
        c2_text = chunks[i + 1]["text"]
        # The end of c1 should appear in c2 (or word suffix/prefix overlap)
        words_c1 = c1_text.split()
        words_c2 = c2_text.split()
        overlap_found = any(w in words_c2[:5] for w in words_c1[-5:])
        assert overlap_found, f"Expected overlap between chunk {i} and chunk {i+1}"


def test_short_page_gives_exactly_one_chunk() -> None:
    """Verify that a page with text shorter than chunk_size produces exactly one chunk."""
    short_text = "This is a brief sentence well under chunk limit."
    pages = [{"page_number": 1, "text": short_text}]

    chunks = chunk_pages(pages, file_name="short.pdf", chunk_size=800, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["text"] == short_text
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["file_name"] == "short.pdf"


def test_empty_or_whitespace_chunks_are_skipped() -> None:
    """Verify that empty pages or whitespace-only sections do not produce empty chunks."""
    pages = [
        {"page_number": 1, "text": "Valid content."},
        {"page_number": 2, "text": "   \n\t   "},
        {"page_number": 3, "text": ""},
    ]

    chunks = chunk_pages(pages, file_name="empty_test.pdf", chunk_size=800, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
