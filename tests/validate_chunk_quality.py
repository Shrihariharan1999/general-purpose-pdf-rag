"""Validate chunk token lengths and inspect financial table chunks."""

import json
from pathlib import Path
from transformers import AutoTokenizer

CHUNK_FILE = Path(
    "data/processed/chunks/"
    "doc_a7a0d8261a0923404fc45446_markdown_tables/"
    "chunks.jsonl"
)
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_text(record):
    """Return the chunk text from the available record fields."""
    metadata = record.get("metadata", {})

    for container in (record, metadata):
        for key in ("text", "content", "chunk_text"):
            value = container.get(key)
            if isinstance(value, str):
                return value

    return ""


def get_pages(record):
    """Extract page numbers from chunk metadata."""
    metadata = record.get("metadata", {})

    for container in (record, metadata):
        for key in ("page_numbers", "pages"):
            value = container.get(key)

            if isinstance(value, list):
                return [
                    int(page)
                    for page in value
                    if str(page).isdigit()
                ]

            if isinstance(value, int):
                return [value]

        for key in ("page_number", "page_no"):
            value = container.get(key)
            if isinstance(value, int):
                return [value]

    return []


def main():
    """Measure token counts and inspect table chunks from page 4."""
    if not CHUNK_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNK_FILE}")

    with CHUNK_FILE.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Measure long chunks without applying the tokenizer's 512-token limit.
    tokenizer.model_max_length = 1_000_000

    lengths = []

    for record in records:
        text = get_text(record)
        token_count = len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )
        lengths.append((token_count, record, text))

    lengths.sort(key=lambda item: item[0], reverse=True)

    print(f"Total chunks: {len(records)}")

    if lengths:
        print(f"Longest chunk: {lengths[0][0]} tokens")

    print(
        "Chunks over 512 tokens: "
        f"{sum(count > 512 for count, _, _ in lengths)}"
    )

    print("\n--- Five longest chunks ---")

    for count, record, _ in lengths[:5]:
        chunk_id = record.get(
            "chunk_id",
            record.get("id", "unknown"),
        )
        print(
            f"{chunk_id}: {count} tokens; "
            f"pages={get_pages(record)}"
        )

    print("\n--- Table-like chunks from PDF page 4 ---")
    found = False

    for _, record, text in lengths:
        if 4 in get_pages(record) and (
            "|" in text or "table" in text.lower()
        ):
            found = True
            chunk_id = record.get(
                "chunk_id",
                record.get("id", "unknown"),
            )

            print(f"\nChunk: {chunk_id}")
            print(f"Pages: {get_pages(record)}")
            print(text[:1800])

    if not found:
        print("No table-like chunks found for PDF page 4.")


if __name__ == "__main__":
    main()