"""FastAPI application for AI Chatbot document Q&A."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import tempfile
import threading
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chunker import chunk_pages
from app.config import settings
from app.pdf_reader import PdfReadError, ScannedPdfError, extract_pages
from app.rag import answer_question
from app.vector_store import (
    DocumentNotFoundError,
    add_chunks,
    delete_document,
    get_documents,
    get_indexed_files,
    get_stats,
    is_file_indexed,
    load,
    reset,
    save,
)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved vector index once on startup."""
    logger.info("Starting up: loading FAISS index if present...")
    load()
    yield
    logger.info("Shutting down AI Chatbot application.")


app = FastAPI(
    title="AI Document Chatbot API",
    description="RAG backend powered by LangChain, FAISS, and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request & Response Models
class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Server health status")


class UploadResponse(BaseModel):
    file_name: str = Field(description="Name of the uploaded and indexed PDF")
    pages: int = Field(description="Number of readable pages extracted")
    chunks: int = Field(description="Total chunks generated and stored")


class HistoryTurn(BaseModel):
    question: str = Field(description="User question in the turn")
    answer: str = Field(description="Assistant answer in the turn")


class AskRequest(BaseModel):
    question: str = Field(description="User question to answer against indexed documents")
    history: list[HistoryTurn] | None = Field(default=None, description="Optional recent conversation turns")


class SourceItem(BaseModel):
    file_name: str = Field(description="Source PDF file name")
    page_number: int = Field(description="Page number containing the chunk")


class AskResponse(BaseModel):
    answer: str = Field(description="Generated answer from context")
    sources: list[SourceItem] = Field(description="Deduplicated and sorted sources")
    retrieved: list[SourceItem] = Field(default_factory=list, description="Every chunk retrieved and sent to LLM")
    top_k: int = Field(default=settings.top_k, description="Configured top_k retrieval chunk count")
    response_time_seconds: float = Field(description="Total roundtrip response time in seconds")
    retrieval_seconds: float = Field(description="Context retrieval duration in seconds")
    llm_seconds: float = Field(description="LLM generation duration in seconds")
    rewrite_seconds: float | None = Field(default=None, description="Question rewrite duration in seconds")
    standalone_question: str | None = Field(default=None, description="Rewritten standalone question when history was used")


class StatsResponse(BaseModel):
    documents: int = Field(description="Count of indexed documents")
    pages: int = Field(description="Count of unique indexed pages")
    chunks: int = Field(description="Total chunks in the vector index")
    files: list[str] = Field(description="List of indexed file names")
    top_k: int = Field(default=settings.top_k, description="Configured top_k retrieval chunk count")


class ClearResponse(BaseModel):
    status: str = Field(default="ok", description="Operation status")
    message: str = Field(description="Confirmation message")


class DocumentItem(BaseModel):
    file_name: str = Field(description="Name of the indexed document")
    pages: int = Field(description="Total readable pages indexed")
    chunks: int = Field(description="Total chunks generated for this document")


class DocumentDeleteResponse(BaseModel):
    file_name: str = Field(description="Name of the deleted PDF document")
    chunks_removed: int = Field(description="Total chunks removed from the vector store")
    pages_removed: int = Field(description="Total pages removed")


# Shared thread lock to guarantee safe concurrent modifications to the vector store
store_lock = threading.Lock()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> dict[str, str]:
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse, tags=["Vector Store"])
def stats() -> dict[str, Any]:
    """Return index statistics and list of indexed file names."""
    with store_lock:
        index_stats = get_stats()
        indexed_files = sorted(list(get_indexed_files()))
    return {
        "documents": index_stats.get("documents", 0),
        "pages": index_stats.get("pages", 0),
        "chunks": index_stats.get("chunks", 0),
        "files": indexed_files,
        "top_k": settings.top_k,
    }


@app.get("/documents", response_model=list[DocumentItem], tags=["Documents"])
def list_documents() -> list[dict[str, Any]]:
    """List all indexed documents with page and chunk counts."""
    with store_lock:
        return get_documents()


@app.delete("/documents/{file_name:path}", response_model=DocumentDeleteResponse, tags=["Documents"])
def delete_single_document(file_name: str) -> dict[str, Any]:
    """Delete a single document by its file name (URL-decoded)."""
    decoded_file_name = unquote(file_name).strip()
    if not decoded_file_name:
        raise HTTPException(status_code=400, detail="Invalid or empty file name provided.")

    with store_lock:
        try:
            result = delete_document(decoded_file_name)
        except DocumentNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{decoded_file_name}' not found.",
            )

    return result


@app.post("/clear", response_model=ClearResponse, tags=["Vector Store"])
def clear() -> dict[str, str]:
    """Clear the vector store index and remove persisted disk files."""
    with store_lock:
        reset()
    return {
        "status": "ok",
        "message": "Vector store index has been reset and cleared.",
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload, extract, chunk, and index a PDF file (under 20 MB)."""
    filename = file.filename or "uploaded.pdf"

    # Validate file extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. File must have a .pdf extension.",
        )

    # Validate MIME type
    if file.content_type and "pdf" not in file.content_type.lower() and file.content_type != "application/octet-stream":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Content-Type '{file.content_type}'. Expected 'application/pdf'.",
        )

    # Check for duplicate file
    with store_lock:
        if is_file_indexed(filename):
            raise HTTPException(
                status_code=409,
                detail=f"Document '{filename}' is already indexed. Use /clear first if you wish to re-index it.",
            )

    # Stream file into temporary file and enforce 20 MB size limit
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_path = Path(temp_file.name)
    total_bytes = 0

    try:
        while True:
            chunk = file.file.read(1024 * 1024)  # 1 MB chunk
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds the 20 MB size limit.",
                )
            temp_file.write(chunk)

        temp_file.flush()
        temp_file.close()

        # Process the PDF
        pages = extract_pages(temp_path)
        chunks = chunk_pages(pages, file_name=filename)
        with store_lock:
            add_chunks(chunks)
            save()

        return {
            "file_name": filename,
            "pages": len(pages),
            "chunks": len(chunks),
        }

    except HTTPException:
        raise
    except (PdfReadError, ScannedPdfError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        msg = str(exc).upper()
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(
                status_code=429,
                detail="The AI embedding service is currently rate limited. Please try again shortly.",
            )
        elif "503" in msg or "UNAVAILABLE" in msg:
            raise HTTPException(
                status_code=503,
                detail="The AI embedding service is currently unavailable. Please try again later.",
            )
        logger.exception("Unexpected error while indexing uploaded PDF: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the uploaded document.",
        )
    finally:
        try:
            if not temp_file.closed:
                temp_file.close()
        except Exception:
            pass
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(request: AskRequest) -> dict[str, Any]:
    """Answer a question based on indexed PDF documents with optional history."""
    question = request.question.strip() if request.question else ""
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or contain only whitespace.",
        )

    if request.history is not None and len(request.history) > 6:
        raise HTTPException(
            status_code=422,
            detail="History cannot exceed 6 turns.",
        )

    sanitized_history = None
    if request.history:
        sanitized_history = [
            {"question": turn.question, "answer": turn.answer[:600]}
            for turn in request.history
        ]

    try:
        res = answer_question(question, history=sanitized_history)
        res["top_k"] = settings.top_k
        return res
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc).upper()
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise HTTPException(
                status_code=429,
                detail="The AI generation service is currently rate limited. Please try again shortly.",
            )
        elif "503" in msg or "UNAVAILABLE" in msg:
            raise HTTPException(
                status_code=503,
                detail="The AI generation service is currently unavailable. Please try again later.",
            )
        logger.exception("Unexpected error during question answering: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer.",
        )
