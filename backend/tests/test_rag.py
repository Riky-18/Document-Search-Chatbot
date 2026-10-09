"""Interactive CLI script to query the RAG pipeline against the saved index."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure backend root is on sys.path when running as a direct script
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.rag import answer_question
from app.vector_store import get_stats, load


def _safe_print(text: str = "") -> None:
    """Print safely to standard output without crashing on Windows console encoding."""
    try:
        print(text)
    except (UnicodeEncodeError, OSError):
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            safe_bytes = text.encode(encoding, errors="replace")
            sys.stdout.buffer.write(safe_bytes + b"\n")
            sys.stdout.flush()
        except Exception:
            print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def main() -> int:
    """Load the saved index and run an interactive Q&A loop."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    _safe_print("==================================================")
    _safe_print("AI Chatbot RAG System - Q&A Interactive Console")
    _safe_print("==================================================")

    # 1. Load saved index on startup
    load()
    stats = get_stats()

    if stats.get("chunks", 0) == 0:
        _safe_print("\nWarning: No documents are currently indexed in FAISS.")
        _safe_print("Please index a PDF first using: python tests/test_search.py <path_to_pdf>")
    else:
        _safe_print(
            f"\nIndex loaded: {stats['documents']} document(s), "
            f"{stats['pages']} page(s), {stats['chunks']} chunk(s) ready."
        )

    _safe_print("\nType your question below. Type 'quit' or 'exit' to exit.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            _safe_print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ["quit", "exit", "q"]:
            _safe_print("Goodbye!")
            break

        try:
            result = answer_question(query)
        except Exception as exc:
            _safe_print(f"Error generating answer: {exc}")
            continue

        _safe_print("\nAnswer:")
        _safe_print(result["answer"])

        sources = result.get("sources", [])
        _safe_print("\nSources:")
        if sources:
            for s in sources:
                _safe_print(f"  - {s['file_name']} (Page {s['page_number']})")
        else:
            _safe_print("  None")

        _safe_print(
            f"\nResponse time: {result['response_time_seconds']}s "
            f"(retrieval: {result['retrieval_seconds']}s, llm: {result['llm_seconds']}s)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
