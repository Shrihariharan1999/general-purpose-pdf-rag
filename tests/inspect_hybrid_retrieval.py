
"""Compare vector search with hybrid vector and BM25 retrieval."""

import json
import re
from pathlib import Path
from collections import defaultdict

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DOCUMENT_ID = "doc_a7a0d8261a0923404fc45446"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "multimodal_pdf_chunks"

CHUNK_FILE = (
    Path("data/processed/chunks")
    / f"{DOCUMENT_ID}_typeaware"
    / "chunks.jsonl"
)
VECTOR_DB_DIR = Path("data/processed/vectorstore/chroma")

QUERIES = [
    "What was Apple's net income for the nine months ended June 25, 2022?",
    "Net income nine months ended June 25 2022",
]


def tokenize(text):
    """Tokenize words, dates, and comma-separated financial numbers."""
    return re.findall(
        r"[a-z]+|\d[\d,]*(?:\.\d+)?%?",
        text.casefold(),
    )


def load_chunks():
    """Load chunk text and metadata from the existing JSONL file."""
    if not CHUNK_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNK_FILE}")

    with CHUNK_FILE.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise ValueError("No chunks found.")

    return records


def build_vector_store():
    """Open the existing persistent Chroma collection."""
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )


def hybrid_search(query, records, vector_store, top_k=10):
    """Fuse vector and BM25 rankings using reciprocal rank fusion."""
    tokenized_corpus = [
        tokenize(record.get("text", ""))
        for record in records
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    lexical_order = sorted(
        range(len(records)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )

    lexical_order = [
        index for index in lexical_order
        if bm25_scores[index] > 0
    ][:50]

    bm25_rank = {
        records[index]["chunk_id"]: rank
        for rank, index in enumerate(lexical_order, start=1)
    }

    # Retrieve semantic candidates from the existing vector database.
    vector_results = vector_store.similarity_search_with_score(
        query,
        k=min(20, len(records)),
        filter={"document_id": DOCUMENT_ID},
    )

    vector_rank = {}
    for rank, (document, _) in enumerate(vector_results, start=1):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            vector_rank[chunk_id] = rank

    # Combine both rankings without mixing distance and BM25 score scales.
    record_by_id = {
        record["chunk_id"]: record
        for record in records
    }
    fused_scores = defaultdict(float)

    for chunk_id, rank in vector_rank.items():
        fused_scores[chunk_id] += 1 / (60 + rank)

    for chunk_id, rank in bm25_rank.items():
        fused_scores[chunk_id] += 1 / (60 + rank)

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )[:top_k]

    results = []

    for chunk_id in ranked_ids:
        record = record_by_id.get(chunk_id)
        if not record:
            continue

        results.append({
            "record": record,
            "rrf_score": fused_scores[chunk_id],
            "vector_rank": vector_rank.get(chunk_id),
            "bm25_rank": bm25_rank.get(chunk_id),
        })

    return results


def main():
    """Compare hybrid retrieval results for financial questions."""
    records = load_chunks()
    vector_store = build_vector_store()

    print(f"Chunks loaded: {len(records)}")
    print(f"Vectors stored: {vector_store._collection.count()}")

    for query in QUERIES:
        results = hybrid_search(
            query,
            records,
            vector_store,
            top_k=10,
        )

        print(f"\n{'=' * 70}")
        print(f"Query: {query}")

        target_found = False

        for rank, result in enumerate(results, start=1):
            record = result["record"]
            content = record.get("text", "")

            is_target = (
                record.get("content_type") == "table"
                and "net income" in content.casefold()
                and "79,082" in content
                and "nine months ended" in content.casefold()
                and "june 25, 2022" in content.casefold()
            )
            target_found |= is_target

            print(
                f"\nRank: {rank}"
                f" | RRF: {result['rrf_score']:.5f}"
                f" | Vector rank: {result['vector_rank']}"
                f" | BM25 rank: {result['bm25_rank']}"
            )
            print(f"Type: {record.get('content_type')}")
            print(f"Pages: {record.get('page_numbers')}")
            print(f"Chunk: {record.get('chunk_id')}")
            print(f"Expected table match: {is_target}")
            print(content[:900])

        print(f"\nExpected page 4 table found in top 10: {target_found}")


if __name__ == "__main__":
    main()
