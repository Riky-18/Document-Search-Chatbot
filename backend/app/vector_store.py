"""Vector store management with FAISS and LangChain."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Callable

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from app.providers import get_embedding_model

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = _BACKEND_DIR / "faiss_index"


def parse_retry_delay(err_msg: str, default_backoff: float) -> float:
    """Parse 'retry in Ns' from error message and return N + 1 seconds, or default_backoff."""
    match = re.search(
        r"retry\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?",
        err_msg,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    return default_backoff


def is_retryable_error(exc: Exception) -> bool:
    """Return True if error is 429, 503, 500 or related transient/rate-limit error."""
    msg = str(exc).upper()
    return any(
        code in msg
        for code in (
            "429",
            "503",
            "500",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL",
            "RATE LIMIT",
            "QUOTA",
        )
    )


class VectorStoreManager:
    """Manages indexing, searching, persistence, and stats for FAISS vector store."""

    def __init__(
        self,
        embedding_model: Embeddings | None = None,
        index_dir: str | Path | None = None,
        batch_size: int = 100,
        max_retries: int = 5,
        base_delay: float = 2.0,
        auto_load: bool = True,
    ) -> None:
        self._embedding_model = embedding_model
        self.index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._store: FAISS | None = None

        if auto_load:
            self.load()

    @property
    def embedding_model(self) -> Embeddings:
        """Return the embedding model, lazily instantiating from providers if needed."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def _execute_with_retry(self, operation: Callable[[], Any], action_desc: str = "request") -> Any:
        """Execute a callable with retry on 429, 503, and 500 errors up to max_retries."""
        for attempt in range(self.max_retries):
            try:
                return operation()
            except Exception as exc:
                if is_retryable_error(exc) and attempt < self.max_retries - 1:
                    default_backoff = self.base_delay * (2 ** attempt)
                    delay = parse_retry_delay(str(exc), default_backoff)
                    delay_repr = int(delay) if float(delay).is_integer() else f"{delay:.1f}"
                    print(f"Rate limited, retrying in {delay_repr} seconds...")
                    logger.warning(
                        "Retryable error during %s (%s). Retrying in %ss (attempt %d/%d)...",
                        action_desc,
                        exc,
                        delay_repr,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Failed %s: %s", action_desc, exc)
                    raise

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Embed and store chunks in batches (up to 100 chunks per call) with metadata.

        Args:
            chunks: List of chunk dicts containing 'text', 'file_name',
                    'page_number', and 'chunk_index'.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        # Attempt to load any existing persisted index before adding if not yet loaded
        if self._store is None:
            self.load()

        total_added = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "file_name": c.get("file_name", "unknown"),
                    "page_number": c.get("page_number", 0),
                    "chunk_index": c.get("chunk_index", 0),
                }
                for c in batch
            ]

            def _add_batch(t=texts, m=metadatas):
                if self._store is None:
                    self._store = FAISS.from_texts(
                        texts=t,
                        embedding=self.embedding_model,
                        metadatas=m,
                    )
                else:
                    self._store.add_texts(
                        texts=t,
                        metadatas=m,
                    )

            self._execute_with_retry(_add_batch, action_desc="embedding batch")
            total_added += len(batch)
            logger.debug("Added batch of %d chunk(s) (total: %d/%d)", len(batch), total_added, len(chunks))

        logger.info("Successfully indexed %d chunk(s)", total_added)
        return total_added

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Search the vector store for chunks matching the query, with retry on rate limits.

        Args:
            query: User search question or query string.
            k: Number of nearest neighbors to retrieve (default: 4).

        Returns:
            List of result dicts: [{"text": str, "metadata": dict, "score": float}]
        """
        if not query or not query.strip():
            return []

        if self._store is None:
            self.load()

        if self._store is None:
            logger.warning("Vector store is empty. No documents indexed.")
            return []

        docs_and_scores = self._execute_with_retry(
            lambda: self._store.similarity_search_with_score(query, k=k),
            action_desc="search query embedding",
        )

        results: list[dict[str, Any]] = []
        for doc, score in docs_and_scores:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })
        return results

    def get_indexed_files(self) -> set[str]:
        """Return the set of unique file names currently indexed in the vector store."""
        if self._store is None:
            self.load()

        if self._store is None:
            return set()

        docstore = getattr(self._store, "docstore", None)
        if not docstore or not hasattr(docstore, "_dict"):
            return set()

        return {
            doc.metadata["file_name"]
            for doc in docstore._dict.values()
            if "file_name" in doc.metadata
        }

    def is_file_indexed(self, file_name: str) -> bool:
        """Check whether a specific file name has already been indexed."""
        return file_name in self.get_indexed_files()

    def get_stats(self) -> dict[str, int]:
        """Return the number of documents, unique pages, and chunks in the vector store."""
        if self._store is None:
            self.load()

        if self._store is None:
            return {"documents": 0, "pages": 0, "chunks": 0}

        docstore = getattr(self._store, "docstore", None)
        if not docstore or not hasattr(docstore, "_dict"):
            return {"documents": 0, "pages": 0, "chunks": 0}

        unique_files: set[str] = set()
        unique_pages: set[tuple[str, int]] = set()
        chunk_count = 0

        for doc in docstore._dict.values():
            chunk_count += 1
            file_name = doc.metadata.get("file_name")
            page_number = doc.metadata.get("page_number")
            if file_name:
                unique_files.add(file_name)
            if file_name and page_number is not None:
                unique_pages.add((file_name, page_number))

        return {
            "documents": len(unique_files),
            "pages": len(unique_pages),
            "chunks": chunk_count,
        }

    def save(self, folder_path: str | Path | None = None) -> bool:
        """Persist the FAISS index and docstore to disk.

        Args:
            folder_path: Target directory (default: self.index_dir).

        Returns:
            True if saved, False if vector store was empty.
        """
        if self._store is None:
            logger.warning("Cannot save: vector store has not been initialized.")
            return False

        target_dir = Path(folder_path) if folder_path else self.index_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(target_dir))
        logger.info("Saved FAISS index to %s", target_dir)
        return True

    def load(self, folder_path: str | Path | None = None) -> bool:
        """Load a persisted FAISS index from disk.

        Args:
            folder_path: Source directory (default: self.index_dir).

        Returns:
            True if loaded successfully, False otherwise.
        """
        target_dir = Path(folder_path) if folder_path else self.index_dir
        index_file = target_dir / "index.faiss"
        pkl_file = target_dir / "index.pkl"

        if not (index_file.exists() and pkl_file.exists()):
            return False

        try:
            self._store = FAISS.load_local(
                str(target_dir),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            logger.info("Successfully loaded FAISS index from %s", target_dir)
            return True
        except Exception as exc:
            logger.warning("Could not load FAISS index from %s: %s", target_dir, exc)
            return False

    def reset(self, folder_path: str | Path | None = None) -> None:
        """Clear the in-memory index and delete persisted index files from disk."""
        self._store = None
        target_dir = Path(folder_path) if folder_path else self.index_dir
        if target_dir.exists():
            for item in target_dir.glob("*"):
                try:
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except OSError as exc:
                    logger.warning("Failed removing %s during reset: %s", item, exc)
            try:
                target_dir.rmdir()
            except OSError:
                pass
        logger.info("Vector store reset complete. Files deleted at %s", target_dir)


# Global default manager singleton
_default_manager: VectorStoreManager | None = None


def get_vector_store_manager() -> VectorStoreManager:
    """Return the global default VectorStoreManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = VectorStoreManager()
    return _default_manager


def add_chunks(chunks: list[dict[str, Any]]) -> int:
    """Add chunks to the default vector store."""
    return get_vector_store_manager().add_chunks(chunks)


def search(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Search the default vector store."""
    return get_vector_store_manager().search(query, k=k)


def get_indexed_files() -> set[str]:
    """Return the set of unique indexed file names."""
    return get_vector_store_manager().get_indexed_files()


def is_file_indexed(file_name: str) -> bool:
    """Check whether a specific file name is already indexed."""
    return get_vector_store_manager().is_file_indexed(file_name)


def get_stats() -> dict[str, int]:
    """Get statistics from the default vector store."""
    return get_vector_store_manager().get_stats()


def save(folder_path: str | Path | None = None) -> bool:
    """Persist the default vector store to disk."""
    return get_vector_store_manager().save(folder_path)


def load(folder_path: str | Path | None = None) -> bool:
    """Load the default vector store from disk."""
    return get_vector_store_manager().load(folder_path)


def reset(folder_path: str | Path | None = None) -> None:
    """Reset and delete files from the default vector store."""
    get_vector_store_manager().reset(folder_path)
