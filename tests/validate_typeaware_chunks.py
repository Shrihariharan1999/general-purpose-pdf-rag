"""Validate text and table chunk sizes before embedding."""

import json
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer

CHUNK_FILE = Path(
    "data/processed/chunks/"
    "doc_a7a0d8261a0923404fc45446_typeaware/"
    "chunks.jsonl"
)
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    """Measure chunk lengths and identify excessively short text chunks."""
    if not CHUNK_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNK_FILE}")

    with CHUNK_FILE.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000

    for record in records:
        text = record.get("text", "")
        record["_tokens"] = len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )

    for chunk_type in ("text", "table"):
        chunks = [
            record for record in records
            if record.get("content_type") == chunk_type
        ]

        lengths = [record["_tokens"] for record in chunks]

        print(f"\n--- {chunk_type.upper()} CHUNKS ---")
        print(f"Count: {len(chunks)}")

        if not lengths:
            continue

        print(f"Average tokens: {mean(lengths):.1f}")
        print(f"Median tokens: {median(lengths):.1f}")
        print(f"Minimum tokens: {min(lengths)}")
        print(f"Maximum tokens: {max(lengths)}")
        print(f"Chunks below 32 tokens: {sum(n < 32 for n in lengths)}")
        print(f"Chunks below 64 tokens: {sum(n < 64 for n in lengths)}")

    text_chunks = sorted(
        [
            record for record in records
            if record.get("content_type") == "text"
        ],
        key=lambda record: record["_tokens"],
    )

    print("\n--- 10 SHORTEST TEXT CHUNKS ---")

    for record in text_chunks[:10]:
        print(
            f"\nTokens: {record['_tokens']}"
            f" | Pages: {record.get('page_numbers', [])}"
            f" | Headings: {record.get('headings', [])}"
        )
        print(record.get("text", "")[:250])

    print("\nValidation complete.")


if __name__ == "__main__":
    main()