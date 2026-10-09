"""Gemini model providers using langchain-google-genai."""

from __future__ import annotations

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedding_model(
    model_name: str | None = None,
    **kwargs: Any,
) -> GoogleGenerativeAIEmbeddings:
    """Return a configured Google Generative AI embeddings model.

    Args:
        model_name: Optional override for the embedding model name.
        **kwargs: Additional parameters passed to GoogleGenerativeAIEmbeddings.

    Returns:
        GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    api_key = settings.gemini_api_key  # Will raise ValueError if missing
    target_model = model_name or settings.embedding_model

    logger.debug("Initializing GoogleGenerativeAIEmbeddings with model: %s", target_model)
    return GoogleGenerativeAIEmbeddings(
        model=target_model,
        google_api_key=api_key,
        **kwargs,
    )


def get_chat_model(
    model_name: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Return a configured Google Generative AI chat model.

    Args:
        model_name: Optional override for the chat model name.
        temperature: Sampling temperature (default: 0.0 for deterministic answers).
        **kwargs: Additional parameters passed to ChatGoogleGenerativeAI.

    Returns:
        ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    api_key = settings.gemini_api_key  # Will raise ValueError if missing
    target_model = model_name or settings.chat_model

    logger.debug("Initializing ChatGoogleGenerativeAI with model: %s", target_model)
    return ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
