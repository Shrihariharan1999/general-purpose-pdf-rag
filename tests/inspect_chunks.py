"""Inspect and validate type-aware chunk records."""

import json
from pathlib import Path

CHUNK_FILE = Path(
    "data/processed/chunks/"
    "doc_a7a0d8261a0923404fc45446_typeaware/"
    "chunks.jsonl"
)


def get_field(record, key, default=None):
    """Read a field from either the record or its metadata."""
    if key in record:
        return record[key]

    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get(key, default)

    return default


def get_text(record):
    """Extract chunk text using supported field names."""
    for key in ("text", "content", "chunk_text"):
        value = get_field(record, key)
        if isinstance(value, str):
            return value

    return ""


def get_pages(record):
    """Extract page numbers from the chunk metadata."""
    for key in ("page_numbers", "pages"):
        value = get_field(record, key)

        if isinstance(value, int):
            return [value]

        if isinstance(value, list):
            return [
                int(page)
                for page in value
                if str(page).isdigit()
            ]

    value = get_field(record, "page_number")
    return [value] if isinstance(value, int) else []


def get_types(record):
    """Return element types or the declared content type."""
    value = get_field(record, "element_types", [])

    if isinstance(value, str):
        value = [value]

    content_type = get_field(record, "content_type")
    if content_type and content_type not in value:
        value.append(content_type)

    return value


def main():
    """Print chunk samples and validate metadata completeness."""
    if not CHUNK_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNK_FILE}")

    with CHUNK_FILE.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    missing_pages = [
        record for record in records if not get_pages(record)
    ]
    empty_text = [
        record for record in records if not get_text(record).strip()
    ]
    table_chunks = [
        record for record in records
        if get_field(record, "content_type") == "table"
    ]
    text_chunks = [
        record for record in records
        if get_field(record, "content_type") == "text"
    ]

    print(f"File: {CHUNK_FILE}")
    print(f"Total chunks: {len(records)}")
    print(f"Text chunks: {len(text_chunks)}")
    print(f"Table chunks: {len(table_chunks)}")
    print(f"Chunks missing page metadata: {len(missing_pages)}")
    print(f"Chunks with empty text: {len(empty_text)}")

    print("\n--- First 3 chunks ---")

    for record in records[:3]:
        print(f"\nID: {get_field(record, 'chunk_id', 'unknown')}")
        print(f"Type: {get_field(record, 'content_type', 'unknown')}")
        print(f"Pages: {get_pages(record)}")
        print(f"Element types: {get_types(record)}")
        print(f"Headings: {get_field(record, 'headings', [])}")
        print(f"Content:\n{get_text(record)[:1200]}")

    page_4_tables = [
        record for record in table_chunks
        if 4 in get_pages(record)
    ]

    print("\n--- Table chunks from PDF page 4 ---")
    print(f"Found: {len(page_4_tables)}")

    for record in page_4_tables:
        print(f"\nID: {get_field(record, 'chunk_id', 'unknown')}")
        print(f"Pages: {get_pages(record)}")
        print(get_text(record)[:2500])

    if missing_pages or empty_text:
        print("\nWARNING: Some chunks need investigation.")
    else:
        print("\nInspection complete: text and page metadata are present.")


if __name__ == "__main__":
    main()