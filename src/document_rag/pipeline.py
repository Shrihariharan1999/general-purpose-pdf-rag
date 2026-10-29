"""End-to-end PDF extraction, chunking, and vector indexing."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

from .config import CHUNKS_DIR, DOCLING_OUTPUT_DIR
from .chunking import chunk_document
from .embeddings import (
    COLLECTION_NAME,
    VECTOR_DB_DIR,
    index_document,
)
from .ingestion import ingest_pdf

ProgressCallback = Callable[[str, float | None], None]
MAX_PDF_BYTES = 100 * 1024 * 1024


def find_indexed_pdf(content_sha256: str) -> dict[str, Any] | None:
    """Find a matching PDF only when its chunks and Chroma vectors exist."""
    for manifest_path in DOCLING_OUTPUT_DIR.glob("*/manifest.json"):
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"Unable to read ingestion manifest {manifest_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if manifest.get("content_sha256") != content_sha256:
            continue

        document_id = str(
            manifest.get("document_id") or manifest_path.parent.name
        )
        chunk_file = (
            CHUNKS_DIR / f"{document_id}_typeaware" / "chunks.jsonl"
        )
        if not chunk_file.is_file():
            return None

        if not VECTOR_DB_DIR.exists():
            return None

        import chromadb

        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            return None

        indexed_vectors = collection.get(
            where={"document_id": document_id},
            include=[],
        )
        vector_count = len(indexed_vectors.get("ids", []))
        if vector_count < 1:
            return None

        with chunk_file.open("r", encoding="utf-8") as file:
            chunk_count = sum(1 for line in file if line.strip())
        return {
            "document_id": document_id,
            "source_file": manifest.get("source_file", document_id),
            "page_count": manifest.get("page_count", 0),
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "manifest": manifest,
        }

    return None


def validate_pdf(pdf_path: Path) -> None:
    """Validate path, size, and PDF signature before expensive processing."""
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.casefold() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path.name}")
    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF is empty: {pdf_path.name}")
    if pdf_path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(
            f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB upload limit."
        )

    with pdf_path.open("rb") as file:
        header = file.read(1024)
    if b"%PDF-" not in header:
        raise ValueError(
            f"{pdf_path.name} does not have a valid PDF file signature."
        )


def process_pdf(
    pdf_path: Path,
    converter: Any = None,
    embedding_model: Any = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the complete PDF-to-Chroma pipeline for one validated PDF."""
    pdf_path = pdf_path.expanduser().resolve()
    validate_pdf(pdf_path)

    from .ingestion import calculate_sha256

    content_sha256 = calculate_sha256(pdf_path)
    existing = find_indexed_pdf(content_sha256)
    if existing is not None:
        if on_progress is not None:
            on_progress("Already indexed; skipped duplicate PDF", 1.0)
        return {
            **existing,
            "already_indexed": True,
        }

    try:
        manifest = ingest_pdf(
            pdf_path,
            converter=converter,
            on_progress=on_progress,
        )
        document_id = str(manifest["document_id"])
        if on_progress is not None:
            on_progress("Creating text, table, and image chunks", 0.76)
        chunk_summary = chunk_document(document_id, on_progress=on_progress)
        if on_progress is not None:
            on_progress("Embedding chunks and indexing in Chroma", 0.85)
        index_document(
            document_id,
            embedding_model=embedding_model,
            on_progress=on_progress,
        )
    except Exception as exc:
        if on_progress is not None:
            on_progress(f"Processing failed: {exc}", None)
        raise

    result = {
        "document_id": document_id,
        "source_file": pdf_path.name,
        "page_count": manifest.get("page_count", 0),
        "chunk_count": chunk_summary.get("total_chunks", 0),
        "vector_count": chunk_summary.get("total_chunks", 0),
        "manifest": manifest,
        "already_indexed": False,
    }
    if on_progress is not None:
        on_progress("Processing complete", 1.0)
    return result
