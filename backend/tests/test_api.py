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
