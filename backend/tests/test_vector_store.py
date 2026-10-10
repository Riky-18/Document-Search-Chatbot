"""Tests for FAISS vector store using offline fake embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from app.vector_store import (
    VectorStoreManager,
    is_retryable_error,
    parse_retry_delay,
)


class FlakyEmbeddings(Embeddings):
    """Embeddings implementation that simulates transient rate limiting."""

    def __init__(self, size: int = 16, error_type: str = "429") -> None:
        self.size = size
        self.error_type = error_type
        self.doc_call_count = 0
        self.query_call_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_call_count += 1
        if self.doc_call_count == 1:
            if self.error_type == "429":
                raise RuntimeError("429 ResourceExhausted: rate limit exceeded, retry in 2s")
            elif self.error_type == "503":
                raise RuntimeError("503 Service Unavailable")
            elif self.error_type == "500":
                raise RuntimeError("500 Internal Server Error")
        return [[0.1] * self.size for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_call_count += 1
        if self.query_call_count == 1:
            if self.error_type == "429":
                raise RuntimeError("429 ResourceExhausted: retry in 3s")
            elif self.error_type == "503":
                raise RuntimeError("503 Service Unavailable")
            elif self.error_type == "500":
                raise RuntimeError("500 Internal Server Error")
        return [0.1] * self.size


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Fixture providing a deterministic offline fake embedding model."""
    return FakeEmbeddings(size=16)


@pytest.fixture
def manager(fake_embeddings: FakeEmbeddings, tmp_path: Path) -> VectorStoreManager:
    """Fixture providing a fresh VectorStoreManager with isolated temp index directory."""
    return VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=tmp_path / "faiss_test_index",
        batch_size=2,
        auto_load=False,
    )


def test_add_chunks_and_search(manager: VectorStoreManager) -> None:
    """Verify adding chunks stores metadata and search returns text, metadata, and score."""
    chunks = [
        {
            "text": "Artificial intelligence algorithms process data.",
            "file_name": "ai.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "text": "Photosynthesis is the process used by plants.",
            "file_name": "biology.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
    ]

    added = manager.add_chunks(chunks)
    assert added == 2

    results = manager.search("intelligence", k=1)
    assert len(results) == 1
    result = results[0]

    assert "text" in result
    assert "metadata" in result
    assert "score" in result
    assert isinstance(result["score"], float)
    assert result["metadata"]["file_name"] in ["ai.pdf", "biology.pdf"]
    assert "chunk_index" in result["metadata"]
    assert "page_number" in result["metadata"]


def test_get_stats(manager: VectorStoreManager) -> None:
    """Verify statistics correctly report document, page, and chunk counts."""
    stats = manager.get_stats()
    assert stats == {"documents": 0, "pages": 0, "chunks": 0}

    chunks = [
        {"text": "Doc 1 Page 1 Chunk 0", "file_name": "doc1.pdf", "page_number": 1, "chunk_index": 0},
        {"text": "Doc 1 Page 1 Chunk 1", "file_name": "doc1.pdf", "page_number": 1, "chunk_index": 1},
        {"text": "Doc 1 Page 2 Chunk 2", "file_name": "doc1.pdf", "page_number": 2, "chunk_index": 2},
        {"text": "Doc 2 Page 1 Chunk 0", "file_name": "doc2.pdf", "page_number": 1, "chunk_index": 0},
    ]
    manager.add_chunks(chunks)

    stats = manager.get_stats()
    assert stats["documents"] == 2
    assert stats["pages"] == 3
    assert stats["chunks"] == 4


def test_save_and_load(manager: VectorStoreManager, fake_embeddings: FakeEmbeddings, tmp_path: Path) -> None:
    """Verify persistence to disk and reloading into a new manager."""
    index_dir = tmp_path / "persistent_index"
    manager.index_dir = index_dir

    chunks = [
        {"text": "Persistent knowledge chunk.", "file_name": "stored.pdf", "page_number": 1, "chunk_index": 0}
    ]
    manager.add_chunks(chunks)
    assert manager.save() is True

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()

    new_manager = VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=index_dir,
        auto_load=True,
    )

    stats = new_manager.get_stats()
    assert stats["chunks"] == 1
    assert stats["documents"] == 1

    results = new_manager.search("knowledge", k=1)
    assert len(results) == 1
    assert results[0]["text"] == "Persistent knowledge chunk."


def test_reset(manager: VectorStoreManager) -> None:
    """Verify reset clears in-memory state and deletes disk files."""
    chunks = [
        {"text": "Temporary data.", "file_name": "temp.pdf", "page_number": 1, "chunk_index": 0}
    ]
    manager.add_chunks(chunks)
    manager.save()

    assert (manager.index_dir / "index.faiss").exists()

    manager.reset()

    assert manager.get_stats() == {"documents": 0, "pages": 0, "chunks": 0}
    assert not (manager.index_dir / "index.faiss").exists()


def test_search_empty_store_returns_empty_list(manager: VectorStoreManager) -> None:
    """Verify search returns empty list when no documents are indexed."""
    assert manager.search("anything") == []


def test_is_file_indexed_and_skip_logic(tmp_path: Path, fake_embeddings: FakeEmbeddings) -> None:
    """Verify skip-if-indexed check accurately reports indexed files before and after reload."""
    index_dir = tmp_path / "skip_test_index"
    mgr = VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=index_dir,
        auto_load=False,
    )

    assert mgr.is_file_indexed("report.pdf") is False
    assert mgr.get_indexed_files() == set()

    chunks = [
        {"text": "Content of report", "file_name": "report.pdf", "page_number": 1, "chunk_index": 0}
    ]
    mgr.add_chunks(chunks)
    mgr.save()

    assert mgr.is_file_indexed("report.pdf") is True
    assert mgr.is_file_indexed("other.pdf") is False
    assert mgr.get_indexed_files() == {"report.pdf"}

    # Reload from disk into fresh manager
    fresh_mgr = VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=index_dir,
        auto_load=True,
    )
    assert fresh_mgr.is_file_indexed("report.pdf") is True
    assert fresh_mgr.is_file_indexed("other.pdf") is False


def test_rate_limit_retry_add_chunks(tmp_path: Path) -> None:
    """Verify retry logic on add_chunks transient errors (429, 503, 500)."""
    for err_code in ["429", "503", "500"]:
        flaky = FlakyEmbeddings(size=16, error_type=err_code)
        flaky_manager = VectorStoreManager(
            embedding_model=flaky,
            index_dir=tmp_path / f"flaky_index_{err_code}",
            batch_size=2,
            base_delay=0.01,
            auto_load=False,
        )

        chunks = [
            {"text": "Text 1", "file_name": "retry.pdf", "page_number": 1, "chunk_index": 0},
        ]

        added = flaky_manager.add_chunks(chunks)
        assert added == 1
        assert flaky.doc_call_count == 2  # 1 initial failure + 1 retry success


def test_rate_limit_retry_search(tmp_path: Path) -> None:
    """Verify search() uses retry logic on 429, 503, 500 up to 5 attempts."""
    for err_code in ["429", "503", "500"]:
        flaky = FlakyEmbeddings(size=16, error_type=err_code)
        flaky_manager = VectorStoreManager(
            embedding_model=flaky,
            index_dir=tmp_path / f"flaky_search_{err_code}",
            base_delay=0.01,
            auto_load=False,
        )

        chunks = [
            {"text": "Search target text", "file_name": "search.pdf", "page_number": 1, "chunk_index": 0},
        ]
        # First add chunks without flaky error on docs
        flaky.doc_call_count = 1  # doc count >= 1 won't fail
        flaky_manager.add_chunks(chunks)

        # Now search: embed_query call #1 will raise the error, call #2 will succeed
        results = flaky_manager.search("target", k=1)
        assert len(results) == 1
        assert flaky.query_call_count == 2  # 1 failure + 1 retry success


def test_parse_retry_delay_logic() -> None:
    """Verify retry delay parser extracts N + 1 seconds when 'retry in Ns' is present."""
    assert parse_retry_delay("Resource exhausted. Please retry in 3s.", default_backoff=2.0) == 4.0
    assert parse_retry_delay("Rate limit hit: retry in 10 seconds", default_backoff=2.0) == 11.0
    assert parse_retry_delay("quota exceeded, retry after 4.5s", default_backoff=2.0) == 5.5
    # When no explicit retry delay is mentioned, use default backoff
    assert parse_retry_delay("503 Service Unavailable", default_backoff=8.0) == 8.0


def test_is_retryable_error_identification() -> None:
    """Verify retryable error detection for 429, 503, 500, quota, and resource exhaustion."""
    assert is_retryable_error(RuntimeError("429 Too Many Requests")) is True
    assert is_retryable_error(RuntimeError("503 Service Unavailable")) is True
    assert is_retryable_error(RuntimeError("500 Internal Server Error")) is True
    assert is_retryable_error(RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")) is True
    assert is_retryable_error(ValueError("Invalid syntax")) is False


def test_delete_document_success(manager: VectorStoreManager) -> None:
    """Verify delete_document removes matching chunks and updates stats."""
    from app.vector_store import DocumentNotFoundError
    chunks = [
        {"text": "A1", "file_name": "docA.pdf", "page_number": 1, "chunk_index": 0},
        {"text": "A2", "file_name": "docA.pdf", "page_number": 2, "chunk_index": 1},
        {"text": "B1", "file_name": "docB.pdf", "page_number": 1, "chunk_index": 0},
    ]
    manager.add_chunks(chunks)

    res = manager.delete_document("docA.pdf")
    assert res == {"file_name": "docA.pdf", "chunks_removed": 2, "pages_removed": 2}

    stats = manager.get_stats()
    assert stats["documents"] == 1
    assert stats["pages"] == 1
    assert stats["chunks"] == 1
    assert manager.get_indexed_files() == {"docB.pdf"}

    # Deleting an unindexed document raises DocumentNotFoundError
    with pytest.raises(DocumentNotFoundError):
        manager.delete_document("docA.pdf")


def test_delete_document_rebuild_fallback(manager: VectorStoreManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify delete_document falls back to rebuilding index if store.delete raises exception."""
    chunks = [
        {"text": "A1", "file_name": "docA.pdf", "page_number": 1, "chunk_index": 0},
        {"text": "B1", "file_name": "docB.pdf", "page_number": 1, "chunk_index": 0},
    ]
    manager.add_chunks(chunks)
    assert manager._store is not None

    def _failing_delete(*args, **kwargs):
        raise NotImplementedError("Index delete not supported")

    monkeypatch.setattr(manager._store, "delete", _failing_delete)

    res = manager.delete_document("docA.pdf")
    assert res["file_name"] == "docA.pdf"
    assert res["chunks_removed"] == 1

    stats = manager.get_stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert manager.get_indexed_files() == {"docB.pdf"}

