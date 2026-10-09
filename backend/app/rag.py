"""Retrieval-Augmented Generation (RAG) question answering pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.providers import get_chat_model
from app.vector_store import (
    VectorStoreManager,
    get_stats,
    get_vector_store_manager,
    is_retryable_error,
    parse_retry_delay,
    search,
)

logger = logging.getLogger(__name__)

DONT_KNOW_PATTERNS = [
    "don't know",
    "dont know",
    "do not know",
    "i don't have enough information",
    "i do not have enough information",
    "not mentioned in the context",
    "not provided in the context",
    "not found in the context",
    "not contain information",
    "context does not contain",
    "context does not mention",
    "context does not provide",
]


def is_dont_know_response(answer: str) -> bool:
    """Return True if the model's answer indicates it does not know or lacks context."""
    normalized = answer.strip().lower()
    return any(pattern in normalized for pattern in DONT_KNOW_PATTERNS)


def build_rag_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build the strict RAG prompt with numbered context chunks."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page_number = meta.get("page_number", 1)
        text = chunk.get("text", "").strip()
        context_blocks.append(f"[{i}] (File: {file_name}, Page: {page_number})\n{text}")

    context_str = "\n\n".join(context_blocks)
    return (
        "Answer using only the context below. If the answer is not in the context, say you don't know. "
        "Mention nothing that is not in the context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def call_chat_model_with_retry(
    chat_model: Any,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> str:
    """Invoke the chat model with rate-limit and transient error retry logic."""
    for attempt in range(max_retries):
        try:
            response = chat_model.invoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):
                # Flatten potential list of message content fragments
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            return str(content).strip()
        except Exception as exc:
            if is_retryable_error(exc) and attempt < max_retries - 1:
                default_backoff = base_delay * (2 ** attempt)
                delay = parse_retry_delay(str(exc), default_backoff)
                delay_repr = int(delay) if float(delay).is_integer() else f"{delay:.1f}"
                print(f"Rate limited, retrying in {delay_repr} seconds...")
                logger.warning(
                    "Retryable error calling chat model (%s). Retrying in %ss (attempt %d/%d)...",
                    exc,
                    delay_repr,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
            else:
                logger.error("Chat model invocation failed: %s", exc)
                raise


def answer_question(
    question: str,
    chat_model: Any | None = None,
    vector_store_manager: VectorStoreManager | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Generate an answer using only retrieved context chunks.

    Args:
        question: User query string.
        chat_model: Optional chat model override (defaults to providers.get_chat_model()).
        vector_store_manager: Optional VectorStoreManager override.
        top_k: Number of context chunks to retrieve (default: 4).

    Returns:
        dict: {
            "answer": str,
            "sources": [{"file_name": str, "page_number": int}],
            "response_time_seconds": float,
            "retrieval_seconds": float,
            "llm_seconds": float,
        }
    """
    start_time = time.perf_counter()

    # 1. Check if vector store has any indexed documents
    vm = vector_store_manager or get_vector_store_manager()
    stats = vm.get_stats()
    if stats.get("chunks", 0) == 0:
        elapsed = round(time.perf_counter() - start_time, 2)
        return {
            "answer": "No documents are currently indexed. Please upload or index a PDF first.",
            "sources": [],
            "response_time_seconds": elapsed,
            "retrieval_seconds": 0.0,
            "llm_seconds": 0.0,
        }

    # 2. Retrieve relevant context chunks with timing
    search_func = vm.search if vector_store_manager else search
    retrieval_start = time.perf_counter()
    chunks = search_func(question, k=top_k)
    retrieval_seconds = round(time.perf_counter() - retrieval_start, 2)

    if not chunks:
        elapsed = round(time.perf_counter() - start_time, 2)
        return {
            "answer": "I don't know. No relevant information was found in the indexed documents.",
            "sources": [],
            "response_time_seconds": elapsed,
            "retrieval_seconds": retrieval_seconds,
            "llm_seconds": 0.0,
        }

    # Extract, deduplicate, and sort sources
    seen_sources = set()
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page_number = meta.get("page_number", 1)
        key = (file_name, page_number)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"file_name": file_name, "page_number": page_number})

    # Sort sources by file_name and then page_number
    sources.sort(key=lambda s: (s["file_name"], s["page_number"]))

    # 3. Build strict RAG prompt
    prompt = build_rag_prompt(question, chunks)

    # 4. Invoke chat model with retry and timing
    model = chat_model or get_chat_model()
    llm_start = time.perf_counter()
    answer = call_chat_model_with_retry(model, prompt)
    llm_seconds = round(time.perf_counter() - llm_start, 2)

    # 6. If the model says it doesn't know, return an empty sources list
    if is_dont_know_response(answer):
        sources = []

    elapsed = round(time.perf_counter() - start_time, 2)
    return {
        "answer": answer,
        "sources": sources,
        "response_time_seconds": elapsed,
        "retrieval_seconds": retrieval_seconds,
        "llm_seconds": llm_seconds,
    }
