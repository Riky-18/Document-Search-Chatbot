"""Retrieval-Augmented Generation (RAG) question answering pipeline."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.config import settings
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


REWRITE_INSTRUCTION = (
    "Rewrite the follow-up question so it can be understood without the conversation. "
    "Replace pronouns and references like 'that' or 'it' with what they refer to. "
    "If it is already standalone, return it unchanged. Return only the rewritten question."
)


def build_rewrite_prompt(question: str, history: list[dict[str, str]]) -> str:
    """Build the prompt for rewriting follow-up questions using history."""
    conv_lines = []
    for turn in history:
        q = turn.get("question", "").strip()
        a = turn.get("answer", "").strip()
        conv_lines.append(f"Question: {q}\nAnswer: {a}")
    conv_str = "\n\n".join(conv_lines)
    return (
        f"{REWRITE_INSTRUCTION}\n\n"
        f"Conversation so far:\n{conv_str}\n\n"
        f"Follow-up question: {question.strip()}\n"
        "Rewritten question:"
    )


def build_rag_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Build the strict RAG prompt with numbered context chunks and optional history."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page_number = meta.get("page_number", 1)
        text = chunk.get("text", "").strip()
        context_blocks.append(f"[{i}] (File: {file_name}, Page: {page_number})\n{text}")

    context_str = "\n\n".join(context_blocks)

    conversation_block = ""
    if history:
        conv_lines = []
        for turn in history:
            q = turn.get("question", "").strip()
            a = turn.get("answer", "").strip()
            conv_lines.append(f"Question: {q}\nAnswer: {a}")
        conversation_block = "Conversation so far:\n" + "\n\n".join(conv_lines) + "\n\n"

    return (
        "Answer using only the context below. If the answer is not in the context, say you don't know. "
        "Mention nothing that is not in the context. "
        "After your answer, add a final line in exactly this format: USED: followed by the numbers of the context chunks you used, comma separated (for example USED: 1,3). If you don't know the answer, write USED: none.\n\n"
        f"{conversation_block}"
        f"Context:\n{context_str}\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def parse_used_chunks(raw_answer: str, total_chunks: int) -> tuple[str, list[int] | None]:
    """Parse and remove the final 'USED: ...' line from the model response.

    Returns:
        (cleaned_answer, used_indices):
            - cleaned_answer: answer string with the USED line removed
            - used_indices: list of 1-based chunk indices (empty list if 'USED: none'),
              or None if the line is missing or unparseable (triggering fallback).
    """
    lines = raw_answer.splitlines()
    last_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break

    if last_idx == -1:
        return raw_answer, None

    last_line = lines[last_idx].strip()
    match = re.match(r"^USED:\s*(.*)$", last_line, re.IGNORECASE)
    if not match:
        return raw_answer, None

    cleaned_answer = "\n".join(lines[:last_idx]).rstrip()
    raw_val = match.group(1).strip()
    # Strip optional trailing punctuation or brackets e.g. "none." or "[1, 2]"
    raw_val_clean = raw_val.strip(".").strip("[]").strip()

    if raw_val_clean.lower() == "none":
        return cleaned_answer, []

    if not raw_val_clean:
        return cleaned_answer, None

    parts = [p.strip() for p in raw_val_clean.split(",") if p.strip()]
    if not parts:
        return cleaned_answer, None

    used_indices: list[int] = []
    for p in parts:
        try:
            idx = int(p)
        except ValueError:
            return cleaned_answer, None
        if 1 <= idx <= total_chunks:
            if idx not in used_indices:
                used_indices.append(idx)
        else:
            return cleaned_answer, None

    return cleaned_answer, used_indices


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
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate an answer using only retrieved context chunks.

    Args:
        question: User query string.
        chat_model: Optional chat model override (defaults to providers.get_chat_model()).
        vector_store_manager: Optional VectorStoreManager override.
        top_k: Number of context chunks to retrieve (default: settings.top_k).
        history: Optional list of recent turns [{"question": str, "answer": str}].

    Returns:
        dict: {
            "answer": str,
            "sources": [{"file_name": str, "page_number": int}],
            "retrieved": [{"file_name": str, "page_number": int}],
            "response_time_seconds": float,
            "retrieval_seconds": float,
            "llm_seconds": float,
            "rewrite_seconds": float (if history was used),
            "standalone_question": str (if history was used),
        }
    """
    start_time = time.perf_counter()
    effective_top_k = top_k if top_k is not None else settings.top_k
    has_history = bool(history)

    # 1. Check if vector store has any indexed documents
    vm = vector_store_manager or get_vector_store_manager()
    stats = vm.get_stats()
    if stats.get("chunks", 0) == 0:
        elapsed = round(time.perf_counter() - start_time, 2)
        res = {
            "answer": "No documents are currently indexed. Please upload or index a PDF first.",
            "sources": [],
            "retrieved": [],
            "response_time_seconds": elapsed,
            "retrieval_seconds": 0.0,
            "llm_seconds": 0.0,
        }
        if has_history:
            res["standalone_question"] = question
            res["rewrite_seconds"] = 0.0
        return res

    model = chat_model or get_chat_model()

    # 2. When history is non-empty, rewrite follow-up question into standalone question
    standalone_question = question
    rewrite_seconds = 0.0
    if has_history:
        rewrite_prompt = build_rewrite_prompt(question, history)
        rewrite_start = time.perf_counter()
        try:
            raw_rewritten = call_chat_model_with_retry(model, rewrite_prompt)
            cleaned_rewritten = raw_rewritten.strip().strip('"\'')
            if cleaned_rewritten:
                standalone_question = cleaned_rewritten
        except Exception as exc:
            logger.warning(
                "Question rewrite failed (%s). Falling back to original question.", exc
            )
            standalone_question = question
        rewrite_seconds = round(time.perf_counter() - rewrite_start, 2)

    # 3. Retrieve relevant context chunks using rewritten question with timing
    search_func = vm.search if vector_store_manager else search
    retrieval_start = time.perf_counter()
    chunks = search_func(standalone_question, k=effective_top_k)
    retrieval_seconds = round(time.perf_counter() - retrieval_start, 2)

    if not chunks:
        elapsed = round(time.perf_counter() - start_time, 2)
        res = {
            "answer": "I don't know. No relevant information was found in the indexed documents.",
            "sources": [],
            "retrieved": [],
            "response_time_seconds": elapsed,
            "retrieval_seconds": retrieval_seconds,
            "llm_seconds": 0.0,
        }
        if has_history:
            res["standalone_question"] = standalone_question
            res["rewrite_seconds"] = rewrite_seconds
        return res

    # List of EVERY chunk retrieved and sent to the LLM
    retrieved = [
        {
            "file_name": chunk.get("metadata", {}).get("file_name", "unknown"),
            "page_number": chunk.get("metadata", {}).get("page_number", 1),
        }
        for chunk in chunks
    ]

    # 4. Build strict RAG prompt with recent history and context
    prompt = build_rag_prompt(standalone_question, chunks, history=history)

    # 5. Invoke chat model with retry and timing
    llm_start = time.perf_counter()
    raw_answer = call_chat_model_with_retry(model, prompt)
    llm_seconds = round(time.perf_counter() - llm_start, 2)

    # Parse and remove the USED: line from the model response
    clean_answer, used_indices = parse_used_chunks(raw_answer, len(chunks))

    # Build sources only from chunks listed, or fallback to all retrieved chunks
    if used_indices is not None:
        selected_chunks = [chunks[i - 1] for i in used_indices]
    else:
        selected_chunks = chunks

    # Extract, deduplicate, and sort sources
    seen_sources = set()
    sources = []
    for chunk in selected_chunks:
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page_number = meta.get("page_number", 1)
        key = (file_name, page_number)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"file_name": file_name, "page_number": page_number})

    # Sort sources by file_name and then page_number
    sources.sort(key=lambda s: (s["file_name"], s["page_number"]))

    # 6. If the model says it doesn't know, return an empty sources list (retrieved remains intact)
    if is_dont_know_response(clean_answer):
        sources = []

    elapsed = round(time.perf_counter() - start_time, 2)
    response_payload = {
        "answer": clean_answer,
        "sources": sources,
        "retrieved": retrieved,
        "response_time_seconds": elapsed,
        "retrieval_seconds": retrieval_seconds,
        "llm_seconds": llm_seconds,
    }
    if has_history:
        response_payload["standalone_question"] = standalone_question
        response_payload["rewrite_seconds"] = rewrite_seconds
    return response_payload
