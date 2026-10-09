"""Configuration settings loaded from environment variables and .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Search for .env in the backend folder or project root
_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")
load_dotenv(_BACKEND_DIR.parent / ".env")

# Model name constants
DEFAULT_CHAT_MODEL: str = "gemini-2.0-flash"
DEFAULT_EMBEDDING_MODEL: str = "models/text-embedding-004"

# Text chunking defaults
DEFAULT_CHUNK_SIZE: int = 800
DEFAULT_CHUNK_OVERLAP: int = 100

# CORS / Origin defaults
DEFAULT_ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]


def _parse_origins(raw: str | None) -> list[str]:
    """Parse a comma-separated string of allowed origins into a list."""
    if not raw:
        return list(DEFAULT_ALLOWED_ORIGINS)
    origins = [part.strip() for part in raw.split(",") if part.strip()]
    return origins or list(DEFAULT_ALLOWED_ORIGINS)


def _int_env(name: str, default: int) -> int:
    """Read an optional integer environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Dataclass holding backend configuration.

    Importing config and instantiating Settings will not fail if GEMINI_API_KEY
    is missing. A clear error is raised only when `gemini_api_key` is accessed.
    """

    allowed_origins: list[str] = field(
        default_factory=lambda: _parse_origins(
            os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS")
        )
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL).strip()
        or DEFAULT_CHAT_MODEL
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL
    )
    chunk_size: int = field(
        default_factory=lambda: _int_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _int_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    )
    _gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        repr=False,
    )

    @property
    def gemini_api_key(self) -> str:
        """Return the Gemini API key, or raise if not set."""
        if not self._gemini_api_key or not self._gemini_api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in backend/.env "
                "or export it as an environment variable."
            )
        return self._gemini_api_key.strip()

    @property
    def cors_origins(self) -> list[str]:
        """Alias for allowed_origins for CORS middleware compatibility."""
        return self.allowed_origins


def get_settings() -> Settings:
    """Build and return a Settings instance from current environment."""
    return Settings()


# Default singleton settings instance for convenient imports
settings = get_settings()
