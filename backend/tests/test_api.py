"""Pytest test suite for FastAPI backend endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest
from langchain_core.embeddings import FakeEmbeddings

import app.main
from app.main import app
from app.vector_store import VectorStoreManager
from tests.test_pdf_reader import build_simple_pdf


class MockChatModel:
    """Mocked chat model for API tests."""

    def __init__(self, response_text: str = "Test answer from AI.") -> None:
        self.response_text = response_text

    def invoke(self, prompt: str) -> MagicMock:
        mock = MagicMock()
        mock.content = self.response_text
        return mock


@pytest.fixture(autouse=True)
def setup_test_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure isolated vector store and mocked models for every test."""
    test_index_dir = tmp_path / "api_faiss_index"
    fake_embeddings = FakeEmbeddings(size=16)

    # Isolated vector store manager
    isolated_manager = VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=test_index_dir,
        batch_size=10,
        auto_load=False,
    )
    monkeypatch.setattr("app.vector_store._default_manager", isolated_manager)
    monkeypatch.setattr("app.main.get_stats", isolated_manager.get_stats)
    monkeypatch.setattr("app.main.get_indexed_files", isolated_manager.get_indexed_files)
    monkeypatch.setattr("app.main.is_file_indexed", isolated_manager.is_file_indexed)
    monkeypatch.setattr("app.main.add_chunks", isolated_manager.add_chunks)
    monkeypatch.setattr("app.main.save", isolated_manager.save)
    monkeypatch.setattr("app.main.load", isolated_manager.load)
    monkeypatch.setattr("app.main.reset", isolated_manager.reset)
    monkeypatch.setattr("app.main.delete_document", isolated_manager.delete_document)
    monkeypatch.setattr("app.main.get_documents", isolated_manager.get_documents)

    # Mock chat model
    mock_chat = MockChatModel()
    monkeypatch.setattr("app.rag.get_chat_model", lambda: mock_chat)


@pytest.fixture
def client() -> TestClient:
    """Return FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient) -> None:
    """GET /health returns status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_valid_pdf_upload(client: TestClient) -> None:
    """POST /upload successfully extracts, chunks, and indexes a valid PDF."""
    pdf_bytes = build_simple_pdf(["First page text content", "Second page text content"])
    files = {"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == "manual.pdf"
    assert data["pages"] == 2
    assert data["chunks"] == 2


def test_non_pdf_upload_rejected(client: TestClient) -> None:
    """POST /upload rejects non-PDF file extension or content-type with 400."""
    # Invalid extension
    files = {"file": ("notes.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400
    assert "Invalid file type" in resp.json()["detail"]


def test_oversized_file_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /upload rejects file exceeding size limit with 413."""
    # Lower the max file size limit for testing
    monkeypatch.setattr("app.main.MAX_FILE_SIZE_BYTES", 50)
    pdf_bytes = build_simple_pdf(["Sample page content that exceeds 50 bytes"])
    files = {"file": ("large.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    resp = client.post("/upload", files=files)
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


def test_duplicate_upload_rejected(client: TestClient) -> None:
    """POST /upload returns 409 for a file already indexed."""
    pdf_bytes = build_simple_pdf(["Page content"])
    files = {"file": ("unique_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    # First upload succeeds
    resp1 = client.post("/upload", files=files)
    assert resp1.status_code == 200

    # Second upload with same file name returns 409
    files_again = {"file": ("unique_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    resp2 = client.post("/upload", files=files_again)
    assert resp2.status_code == 409
    assert "already indexed" in resp2.json()["detail"]


def test_empty_question_rejected(client: TestClient) -> None:
    """POST /ask rejects empty or whitespace question with 422."""
    resp = client.post("/ask", json={"question": "   \n\t  "})
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()


def test_ask_history_exceeding_six_turns_rejected(client: TestClient) -> None:
    """POST /ask rejects history with more than 6 turns with 422."""
    turns = [{"question": f"Question {i}", "answer": f"Answer {i}"} for i in range(7)]
    resp = client.post("/ask", json={"question": "What is next?", "history": turns})
    assert resp.status_code == 422
    assert "6 turns" in resp.json()["detail"].lower()


def test_ask_with_nothing_indexed(client: TestClient) -> None:
    """POST /ask when nothing is indexed returns message without calling LLM."""
    client.post("/clear")  # Ensure clean slate

    resp = client.post("/ask", json={"question": "What is in the document?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "No documents are currently indexed" in data["answer"]
    assert data["sources"] == []
    assert data["retrieved"] == []
    assert "retrieval_seconds" in data
    assert "llm_seconds" in data


def test_ask_with_indexed_content(client: TestClient) -> None:
    """POST /ask successfully answers using context and returns sources and retrieved chunks."""
    pdf_bytes = build_simple_pdf(["Documentation on python and fastapi framework."])
    files = {"file": ("fastapi_guide.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_resp = client.post("/upload", files=files)
    assert upload_resp.status_code == 200

    ask_resp = client.post("/ask", json={"question": "What framework is discussed?"})
    assert ask_resp.status_code == 200
    data = ask_resp.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["file_name"] == "fastapi_guide.pdf"
    assert data["sources"][0]["page_number"] == 1
    assert len(data["retrieved"]) > 0
    assert data["retrieved"][0]["file_name"] == "fastapi_guide.pdf"
    assert data["retrieved"][0]["page_number"] == 1


def test_stats_and_clear(client: TestClient) -> None:
    """GET /stats reports accurate counts, top_k, and POST /clear resets them."""
    # Upload a document
    pdf_bytes = build_simple_pdf(["Page 1", "Page 2"])
    files = {"file": ("stats_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    client.post("/upload", files=files)

    # Check stats
    stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["documents"] == 1
    assert stats["pages"] == 2
    assert stats["chunks"] == 2
    assert "stats_test.pdf" in stats["files"]
    assert stats["top_k"] == 4

    # Clear
    clear_resp = client.post("/clear")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "ok"

    # Check stats after clear
    empty_stats = client.get("/stats").json()
    assert empty_stats["documents"] == 0
    assert empty_stats["pages"] == 0
    assert empty_stats["chunks"] == 0
    assert empty_stats["files"] == []
    assert empty_stats["top_k"] == 4


def test_delete_document_end_to_end(client: TestClient) -> None:
    """Upload two PDFs, delete one, assert /stats drop, assert chunks never appear in /ask sources."""
    # 1. Upload two PDFs
    pdf1 = build_simple_pdf(["Alpha content on page one.", "Alpha content on page two."])
    pdf2 = build_simple_pdf(["Beta content on single page."])

    resp1 = client.post("/upload", files={"file": ("alpha doc.pdf", io.BytesIO(pdf1), "application/pdf")})
    assert resp1.status_code == 200
    resp2 = client.post("/upload", files={"file": ("beta.pdf", io.BytesIO(pdf2), "application/pdf")})
    assert resp2.status_code == 200

    # 2. Check stats: 2 documents, 3 pages, 3 chunks
    stats = client.get("/stats").json()
    assert stats["documents"] == 2
    assert stats["pages"] == 3
    assert stats["chunks"] == 3
    assert sorted(stats["files"]) == ["alpha doc.pdf", "beta.pdf"]

    # Check GET /documents
    docs_resp = client.get("/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) == 2
    assert {"file_name": "alpha doc.pdf", "pages": 2, "chunks": 2} in docs
    assert {"file_name": "beta.pdf", "pages": 1, "chunks": 1} in docs

    # 3. Delete 'alpha doc.pdf' (with space in name)
    del_resp = client.delete("/documents/alpha%20doc.pdf")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["file_name"] == "alpha doc.pdf"
    assert del_data["chunks_removed"] == 2
    assert del_data["pages_removed"] == 2

    # 4. Check stats: counts drop correctly
    stats_after = client.get("/stats").json()
    assert stats_after["documents"] == 1
    assert stats_after["pages"] == 1
    assert stats_after["chunks"] == 1
    assert stats_after["files"] == ["beta.pdf"]

    # 5. Assert deleted file's chunks never appear in /ask sources or retrieved
    ask_resp = client.post("/ask", json={"question": "What is in alpha content?"})
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    source_files = [s["file_name"] for s in ask_data["sources"]]
    retrieved_files = [r["file_name"] for r in ask_data["retrieved"]]
    assert "alpha doc.pdf" not in source_files
    assert "alpha doc.pdf" not in retrieved_files


def test_delete_unknown_document_returns_404(client: TestClient) -> None:
    """DELETE /documents/{file_name} returns 404 if file is not indexed."""
    resp = client.delete("/documents/non_existent.pdf")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_delete_last_document_leaves_clean_empty_state(client: TestClient) -> None:
    """Deleting the last file leaves the store in the same clean empty state as /clear."""
    pdf = build_simple_pdf(["Sole document text."])
    client.post("/upload", files={"file": ("sole.pdf", io.BytesIO(pdf), "application/pdf")})

    del_resp = client.delete("/documents/sole.pdf")
    assert del_resp.status_code == 200

    # Stats show 0 documents
    stats = client.get("/stats").json()
    assert stats["documents"] == 0
    assert stats["pages"] == 0
    assert stats["chunks"] == 0
    assert stats["files"] == []

    # Documents list is empty
    assert client.get("/documents").json() == []

    # /ask returns the exact 'nothing indexed' response
    ask_resp = client.post("/ask", json={"question": "Any question?"})
    assert ask_resp.status_code == 200
    assert "No documents are currently indexed" in ask_resp.json()["answer"]
    assert ask_resp.json()["sources"] == []


def test_index_survives_reload_after_deletion(client: TestClient, tmp_path: Path) -> None:
    """Index survives a reload from disk after a document deletion."""
    import app.vector_store
    pdf1 = build_simple_pdf(["First doc text."])
    pdf2 = build_simple_pdf(["Second doc text."])

    client.post("/upload", files={"file": ("doc1.pdf", io.BytesIO(pdf1), "application/pdf")})
    client.post("/upload", files={"file": ("doc2.pdf", io.BytesIO(pdf2), "application/pdf")})

    client.delete("/documents/doc1.pdf")

    # Force a fresh load from disk using the isolated manager's folder
    mgr = app.vector_store._default_manager
    assert mgr is not None
    loaded = mgr.load()
    assert loaded is True

    stats = mgr.get_stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 1
    assert mgr.get_indexed_files() == {"doc2.pdf"}
