"""Create local embeddings and store them in persistent Chroma."""

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHUNKS_DIR


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "multimodal_pdf_chunks"
VECTOR_DB_DIR = CHUNKS_DIR.parent / "vectorstore" / "chroma"
BATCH_SIZE = 32


def load_chunks(document_id):
    """Load and validate the type-aware JSONL chunks."""
    chunk_file = (
        CHUNKS_DIR / f"{document_id}_typeaware" / "chunks.jsonl"
    )

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_file}"
        )

    records = []

    with chunk_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if record.get("document_id") != document_id:
                raise ValueError(
                    f"Document ID mismatch on line {line_number}"
                )

            if not str(record.get("text", "")).strip():
                continue

            if not record.get("chunk_id"):
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError("No valid chunks were found.")

    chunk_ids = [record["chunk_id"] for record in records]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    print(f"[Embedding] Chunks loaded: {len(records)}")
    return records


@lru_cache(maxsize=1)
def create_embedding_model():
    """Initialize the local sentence-transformer embedding model."""
    print(f"[Embedding] Loading model: {MODEL_NAME}")

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 16,
            "normalize_embeddings": True,
        },
        show_progress=True,
    )


def build_metadata(record):
    """Convert chunk metadata into Chroma-compatible scalar values."""
    pages = record.get("page_numbers", [])
    pages = sorted({
        int(page) for page in pages
        if str(page).isdigit()
    })

    metadata = {
        "document_id": str(record["document_id"]),
        "document_name": str(
            record.get("document_name") or record["document_id"]
        ),
        "chunk_id": str(record["chunk_id"]),
        "content_type": str(record.get("content_type", "text")),
        "page_number": pages[0] if pages else 0,
        "page_start": pages[0] if pages else 0,
        "page_end": pages[-1] if pages else 0,
        "page_numbers": ",".join(map(str, pages)),
        "page_numbers_json": json.dumps(pages),
        "headings": " > ".join(record.get("headings", [])),
        "element_types": ", ".join(
            record.get("element_types", [])
        ),
        "source_regions_json": json.dumps(
            record.get("source_regions", []),
            ensure_ascii=False,
            default=str,
        ),
        "page_images_json": json.dumps(
            record.get("page_images", []),
            ensure_ascii=False,
            default=str,
        ),
        "visual_assets_json": json.dumps(
            record.get("visual_assets", []),
            ensure_ascii=False,
            default=str,
        ),
    }

    # Add optional fields only when they contain usable values.
    for key in (
        "token_count",
        "table_number",
        "table_part",
        "image_number",
    ):
        value = record.get(key)

        if isinstance(value, int):
            metadata[key] = value

    for key in ("caption", "image_uri"):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            metadata[key] = value

    return metadata


def prepare_documents(records):
    """Convert chunk records into LangChain Documents with stable IDs."""
    documents = []
    ids = []

    for record in records:
        text = str(record["text"]).strip()

        documents.append(
            Document(
                page_content=text,
                metadata=build_metadata(record),
            )
        )
        ids.append(str(record["chunk_id"]))

    return documents, ids


def create_vector_store(embedding_model):
    """Open the persistent Chroma collection."""
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def replace_document_vectors(
    vector_store,
    document_id,
    documents,
    ids,
    on_progress: Callable[[str, float | None], None] | None = None,
):
    """Replace vectors for one document without deleting other documents."""
    existing = vector_store.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    existing_ids = existing.get("ids", [])

    if existing_ids:
        print(
            f"[Embedding] Replacing existing vectors: "
            f"{len(existing_ids)}"
        )
        vector_store.delete(ids=existing_ids)

    total = len(documents)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        vector_store.add_documents(
            documents=documents[start:end],
            ids=ids[start:end],
        )
        if on_progress is not None:
            on_progress(
                f"Embedding and indexing chunks: {end}/{total}",
                0.84 + 0.15 * end / total,
            )

        print(
            f"\r[Embedding] Indexed {end}/{total} chunks",
            end="",
            flush=True,
        )

    print()


def test_retrieval(vector_store, document_id, query):
    """Smoke-test semantic retrieval using content from this document."""

    print(f"\n[Retrieval Test] Query: {query}")

    results = vector_store.similarity_search_with_score(
        query,
        k=3,
        filter={"document_id": document_id},
    )

    if not results:
        print("No matching chunks were retrieved.")
        return

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\nResult {rank}")
        print(f"Distance: {distance:.4f}")
        print(f"Type: {metadata.get('content_type')}")
        print(f"Pages: {metadata.get('page_numbers')}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Content:\n{document.page_content[:700]}")


def index_document(
    document_id,
    embedding_model=None,
    on_progress: Callable[[str, float | None], None] | None = None,
):
    """Embed the document's chunks and persist them in Chroma."""
    if on_progress is not None:
        on_progress("Loading document chunks for indexing", 0.84)
    records = load_chunks(document_id)
    documents, ids = prepare_documents(records)

    if embedding_model is None:
        if on_progress is not None:
            on_progress("Loading local embedding model", 0.85)
        embedding_model = create_embedding_model()

    # Confirm the model works and determine its vector dimensions.
    test_vector = embedding_model.embed_query(
        "Embedding model initialization check"
    )
    print(f"[Embedding] Vector dimensions: {len(test_vector)}")

    vector_store = create_vector_store(embedding_model)

    print(f"[Embedding] Database: {VECTOR_DB_DIR}")
    print(f"[Embedding] Collection: {COLLECTION_NAME}")

    replace_document_vectors(
        vector_store,
        document_id,
        documents,
        ids,
        on_progress=on_progress,
    )

    count = vector_store._collection.count()
    print(f"\n[Embedding] Total vectors in collection: {count}")
    print("[Embedding] Indexing completed successfully.")
    if on_progress is not None:
        on_progress("Indexing complete", 1.0)

    test_retrieval(
        vector_store,
        document_id,
        records[0]["text"][:300],
    )


def main():
    """Parse the document ID and run the embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="Embed document chunks and store them in Chroma."
    )
    parser.add_argument(
        "document_id",
        help="Document ID produced by ingestion",
    )
    args = parser.parse_args()

    index_document(args.document_id)


if __name__ == "__main__":
    main()