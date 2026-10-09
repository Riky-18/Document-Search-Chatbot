"""Pytest unit tests for the RAG answer generation pipeline using fake models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import FakeEmbeddings

from app.rag import (
    answer_question,
    build_rag_prompt,
    call_chat_model_with_retry,
    is_dont_know_response,
)
from app.vector_store import VectorStoreManager


class FakeChatModel:
    """Deterministic offline fake chat model for unit testing."""

    def __init__(self, response_text: str = "This is a test answer.") -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt = ""

    def invoke(self, prompt: str) -> MagicMock:
        self.call_count += 1
        self.last_prompt = prompt
        mock_response = MagicMock()
        mock_response.content = self.response_text
        return mock_response


class FlakyChatModel:
    """Chat model that simulates a transient rate limit error before succeeding."""

    def __init__(self, final_text: str = "Recovered answer.") -> None:
        self.final_text = final_text
        self.call_count = 0

    def invoke(self, prompt: str) -> MagicMock:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("429 ResourceExhausted: rate limit exceeded, retry in 1s")
        mock_response = MagicMock()
        mock_response.content = self.final_text
        return mock_response


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=16)


@pytest.fixture
def empty_manager(fake_embeddings: FakeEmbeddings, tmp_path: Path) -> VectorStoreManager:
    return VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=tmp_path / "empty_index",
        auto_load=False,
    )


@pytest.fixture
def populated_manager(fake_embeddings: FakeEmbeddings, tmp_path: Path) -> VectorStoreManager:
    mgr = VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=tmp_path / "populated_index",
        auto_load=False,
    )
    chunks = [
        {
            "text": "Artificial Intelligence is transformative.",
            "file_name": "ai_report.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "text": "Machine Learning is a subset of AI.",
            "file_name": "ai_report.pdf",
            "page_number": 2,
            "chunk_index": 1,
        },
        {
            "text": "Neural networks process layered representations.",
            "file_name": "deep_learning.pdf",
            "page_number": 5,
            "chunk_index": 0,
        },
    ]
    mgr.add_chunks(chunks)
    return mgr


def test_nothing_indexed(empty_manager: VectorStoreManager) -> None:
    """Verify that nothing indexed returns a clear message without calling the LLM."""
    chat_model = FakeChatModel()
    result = answer_question(
        "What is AI?",
        chat_model=chat_model,
        vector_store_manager=empty_manager,
    )

    assert chat_model.call_count == 0  # LLM must not be called
    assert "No documents are currently indexed" in result["answer"]
    assert result["sources"] == []
    assert result["retrieved"] == []
    assert isinstance(result["response_time_seconds"], float)
    assert result["retrieval_seconds"] == 0.0
    assert result["llm_seconds"] == 0.0


def test_normal_answer_with_sources(populated_manager: VectorStoreManager) -> None:
    """Verify a normal answer returns the response text, sorted sources, and response time."""
    chat_model = FakeChatModel("AI is transformative and includes machine learning.")
    result = answer_question(
        "Explain AI",
        chat_model=chat_model,
        vector_store_manager=populated_manager,
    )

    assert chat_model.call_count == 1
    assert result["answer"] == "AI is transformative and includes machine learning."
    assert len(result["sources"]) > 0
    assert len(result["retrieved"]) > 0

    # Ensure sources have file_name and page_number
    for s in result["sources"]:
        assert "file_name" in s
        assert "page_number" in s

    # Ensure retrieved chunks have file_name and page_number
    for r in result["retrieved"]:
        assert "file_name" in r
        assert "page_number" in r

    # Ensure sources are sorted
    sorted_sources = sorted(result["sources"], key=lambda x: (x["file_name"], x["page_number"]))
    assert result["sources"] == sorted_sources

    # Check rounded time and timing breakdown
    assert round(result["response_time_seconds"], 2) == result["response_time_seconds"]
    assert "retrieval_seconds" in result
    assert "llm_seconds" in result
    assert isinstance(result["retrieval_seconds"], float)
    assert isinstance(result["llm_seconds"], float)


def test_duplicate_sources_removed(fake_embeddings: FakeEmbeddings, tmp_path: Path) -> None:
    """Verify that multiple chunks from the same page produce only one source entry."""
    mgr = VectorStoreManager(
        embedding_model=fake_embeddings,
        index_dir=tmp_path / "dup_index",
        auto_load=False,
    )
    # Add 3 chunks from page 1 of doc.pdf and 1 chunk from page 2
    chunks = [
        {"text": "Chunk 0 on page 1", "file_name": "doc.pdf", "page_number": 1, "chunk_index": 0},
        {"text": "Chunk 1 on page 1", "file_name": "doc.pdf", "page_number": 1, "chunk_index": 1},
        {"text": "Chunk 2 on page 1", "file_name": "doc.pdf", "page_number": 1, "chunk_index": 2},
        {"text": "Chunk 3 on page 2", "file_name": "doc.pdf", "page_number": 2, "chunk_index": 3},
    ]
    mgr.add_chunks(chunks)

    chat_model = FakeChatModel("Answer based on doc.pdf.")
    result = answer_question(
        "Summarize doc",
        chat_model=chat_model,
        vector_store_manager=mgr,
        top_k=4,
    )

    # Page 1 had 3 chunks, but should appear only once in sources
    assert result["sources"] == [
        {"file_name": "doc.pdf", "page_number": 1},
        {"file_name": "doc.pdf", "page_number": 2},
    ]
    # retrieved retains all 4 chunks
    assert len(result["retrieved"]) == 4


def test_dont_know_returns_empty_sources(populated_manager: VectorStoreManager) -> None:
    """Verify that if model doesn't know, sources is empty but retrieved keeps all chunks."""
    chat_model = FakeChatModel("I don't know. The provided documents do not contain this information.")
    result = answer_question(
        "Who founded the company?",
        chat_model=chat_model,
        vector_store_manager=populated_manager,
    )

    assert "I don't know" in result["answer"]
    assert result["sources"] == []
    # retrieved keeps the chunks sent to LLM even on 'I don't know'
    assert len(result["retrieved"]) > 0
    assert all("file_name" in r and "page_number" in r for r in result["retrieved"])


def test_top_k_configurable(populated_manager: VectorStoreManager) -> None:
    """Verify top_k retrieval count is configurable."""
    chat_model = FakeChatModel("Sample response.")
    # Request top_k = 2
    result_k2 = answer_question(
        "AI topics",
        chat_model=chat_model,
        vector_store_manager=populated_manager,
        top_k=2,
    )
    assert len(result_k2["retrieved"]) == 2

    # Request default (populated_manager has 3 chunks total)
    result_default = answer_question(
        "AI topics",
        chat_model=chat_model,
        vector_store_manager=populated_manager,
    )
    assert len(result_default["retrieved"]) == 3


def test_rag_prompt_structure() -> None:
    """Verify prompt adheres to the exact requested structure."""
    chunks = [
        {
            "text": "First chunk content.",
            "metadata": {"file_name": "guide.pdf", "page_number": 3},
        },
        {
            "text": "Second chunk content.",
            "metadata": {"file_name": "manual.pdf", "page_number": 1},
        },
    ]
    prompt = build_rag_prompt("How to operate?", chunks)

    expected_prefix = (
        "Answer using only the context below. If the answer is not in the context, say you don't know. "
        "Mention nothing that is not in the context."
    )
    assert prompt.startswith(expected_prefix)
    assert "[1] (File: guide.pdf, Page: 3)" in prompt
    assert "First chunk content." in prompt
    assert "[2] (File: manual.pdf, Page: 1)" in prompt
    assert "Second chunk content." in prompt
    assert "Question: How to operate?" in prompt


def test_chat_model_rate_limit_retry() -> None:
    """Verify retry logic on transient chat model errors."""
    flaky = FlakyChatModel("Final answer.")
    answer = call_chat_model_with_retry(flaky, "test prompt", base_delay=0.01)

    assert answer == "Final answer."
    assert flaky.call_count == 2


def test_chat_model_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_chat_model sets thinking_level='minimal' and max_output_tokens=500."""
    from app.config import (
        DEFAULT_CHAT_MODEL,
        DEFAULT_MAX_OUTPUT_TOKENS,
        DEFAULT_THINKING_LEVEL,
        get_settings,
    )
    from app.providers import get_chat_model

    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    monkeypatch.setattr("app.providers.settings", get_settings())
    model = get_chat_model()

    assert model.model == "gemini-3.1-flash-lite"
    assert DEFAULT_CHAT_MODEL == "gemini-3.1-flash-lite"
    assert DEFAULT_THINKING_LEVEL == "minimal"
    assert DEFAULT_MAX_OUTPUT_TOKENS == 500
    assert model.thinking_level == "minimal"
    assert model.max_output_tokens == 500
