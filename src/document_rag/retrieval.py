"""General hybrid retrieval for PDF document chunks."""

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from .config import (
    CHUNKS_DIR,
    DOCLING_OUTPUT_DIR,
    RERANKER_CANDIDATES,
    RERANKER_ENABLED,
    RERANKER_MODEL,
)
from .embeddings import create_embedding_model


COLLECTION_NAME = "multimodal_pdf_chunks"
VECTOR_DB_DIR = CHUNKS_DIR.parent / "vectorstore" / "chroma"

RRF_K = 60
VECTOR_CANDIDATES = 30
KEYWORD_CANDIDATES = 50


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    """Load and retain one cross-encoder per configured model."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def warm_retrieval_models() -> dict[str, object]:
    """Load the local models before the first user query."""
    models = {"embeddings": create_embedding_model()}
    if RERANKER_ENABLED:
        models["reranker"] = _load_reranker(RERANKER_MODEL)
    return models


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "did", "do", "does", "for", "from",
    "give", "how", "i", "is", "it", "me", "of", "on",
    "please", "show", "tell", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "about",
}


def tokenize(text):
    """Tokenize text while preserving useful words and numeric values."""
    return re.findall(
        r"[a-z]+|\d[\d,]*(?:\.\d+)?%?",
        text.casefold(),
    )


def normalize_query(query):
    """Normalize query terms for keyword retrieval."""
    tokens = [
        token for token in tokenize(query)
        if token not in STOP_WORDS
    ]
    return " ".join(tokens)


def build_query_variants(query):
    """Create simple variants without assuming a document subject."""
    variants = [query.strip()]
    normalized = normalize_query(query)

    if normalized and normalized.casefold() not in {
        item.casefold() for item in variants
    }:
        variants.append(normalized)

    # Normalize possessives without removing the original entity name.
    plain = re.sub(
        r"\b([a-z][a-z0-9-]*)['’]s\b",
        r"\1",
        query,
        flags=re.IGNORECASE,
    )
    normalized_plain = normalize_query(plain)

    if normalized_plain and normalized_plain.casefold() not in {
        item.casefold() for item in variants
    }:
        variants.append(normalized_plain)

    return variants


def load_chunks(document_id):
    """Load chunk records from the selected PDF."""
    path = (
        CHUNKS_DIR / f"{document_id}_typeaware" / "chunks.jsonl"
    )

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise ValueError(f"No chunks found for document {document_id}.")

    manifest_path = (
        DOCLING_OUTPUT_DIR / document_id / "manifest.json"
    )
    document_name = document_id
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        document_name = str(manifest.get("source_file") or document_id)

    for record in records:
        if not record.get("chunk_id") or not record.get("text"):
            raise ValueError("A chunk is missing its ID or text.")
        record.setdefault("document_name", document_name)

    return records


def parse_markdown_table(text):
    """Extract table headers and data rows from a Markdown chunk."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("|")
    ]

    if len(lines) < 2:
        return [], []

    def parse_row(line):
        """Split a Markdown table row into cells."""
        return [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

    headers = parse_row(lines[0])
    rows = []

    for line in lines[1:]:
        cells = parse_row(line)

        # Ignore the Markdown header separator.
        if cells and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
            for cell in cells
        ):
            continue

        rows.append(cells)

    return headers, rows


def query_row_phrases(query, min_length=2, max_length=5):
    """Build generic multiword phrases from meaningful query tokens."""
    tokens = [
        token for token in tokenize(query)
        if token not in STOP_WORDS
    ]
    phrases = set()

    for length in range(min_length, min(max_length, len(tokens)) + 1):
        for start in range(len(tokens) - length + 1):
            phrases.add(tuple(tokens[start:start + length]))

    return phrases


def row_contains_phrase(row_label, phrases):
    """Return whether a row label contains a complete query phrase."""
    row_tokens = tokenize(row_label)

    for phrase in phrases:
        phrase_length = len(phrase)

        for start in range(len(row_tokens) - phrase_length + 1):
            if tuple(row_tokens[start:start + phrase_length]) == phrase:
                return True

    return False


def table_has_query_phrase(record, phrases):
    """Check whether any first-column row label matches a query phrase."""
    if record.get("content_type") != "table":
        return False

    _, rows = parse_markdown_table(record.get("text", ""))

    return any(
        row and row_contains_phrase(row[0], phrases)
        for row in rows
    )


def table_match_features(record, query, phrases=None, prefer_phrases=False):
    """Score table rows and headers without rewarding partial word matches."""
    empty = {
        "row_match": False,
        "header_match": False,
        "bonus": 0.0,
    }

    if record.get("content_type") != "table":
        return empty

    headers, rows = parse_markdown_table(record.get("text", ""))

    if not headers or not rows:
        return empty

    query_terms = {
        token for token in tokenize(query)
        if token not in STOP_WORDS
    }

    if not query_terms:
        return empty

    phrases = phrases or query_row_phrases(query)

    if prefer_phrases:
        # Prefer complete phrases whenever a query phrase matches a row
        # somewhere in the selected document.
        row_match = any(
            row and row_contains_phrase(row[0], phrases)
            for row in rows
        )
    else:
        # Require stronger evidence for longer natural-language queries.
        minimum_overlap = 1 if len(query_terms) <= 2 else 2

        row_match = any(
            row
            and len(
                query_terms.intersection(tokenize(row[0]))
            ) >= minimum_overlap
            for row in rows
        )

    header_terms = set(tokenize(" ".join(headers)))
    header_overlap = query_terms.intersection(header_terms)

    query_has_numeric_detail = any(
        re.search(r"\d", term) for term in query_terms
    )

    header_match = (
        len(header_overlap) >= 2
        or (
            query_has_numeric_detail
            and bool(header_overlap)
        )
    )

    # Header evidence helps only after a row label matches.
    bonus = 0.0

    if row_match:
        bonus += 0.15

        if header_match:
            bonus += 0.08

    return {
        "row_match": row_match,
        "header_match": header_match,
        "bonus": bonus,
    }


class HybridRetriever:
    """Retrieve relevant chunks through vector, keyword, and table matching."""

    def __init__(self, document_id, reranker=None):
        """Initialize the selected document corpus and vector search."""
        self.document_id = document_id
        self.reranker = reranker
        self.reranker_error = None
        self.records = load_chunks(document_id)

        self.records_by_id = {
            record["chunk_id"]: record
            for record in self.records
        }

        self.bm25 = BM25Okapi([
            tokenize(record.get("text", ""))
            for record in self.records
        ])

        self.vector_store = None
        self.vector_error = None

        try:
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=create_embedding_model(),
                persist_directory=str(VECTOR_DB_DIR),
            )

            vector_count = self.vector_store._collection.count()
            print(
                f"[Retrieval] Vectors stored: {vector_count}"
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.vector_error = str(exc)
            self.vector_store = None
            print(
                "[Retrieval] Vector store unavailable; falling back to "
                "BM25 and table heuristics."
            )

        print(f"[Retrieval] Chunks loaded: {len(self.records)}")

    def _get_reranker(self):
        """Load the configured cross-encoder once, when reranking is needed."""
        if self.reranker is None:
            self.reranker = _load_reranker(RERANKER_MODEL)
        return self.reranker

    def search(self, query, top_k=5):
        """Combine semantic, keyword, and table-aware ranking signals."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        fused_scores = defaultdict(float)
        variants = build_query_variants(query)

        for variant in variants:
            if self.vector_store is not None:
                try:
                    vector_results = (
                        self.vector_store.similarity_search_with_score(
                            variant,
                            k=min(VECTOR_CANDIDATES, len(self.records)),
                            filter={"document_id": self.document_id},
                        )
                    )

                    for rank, (document, _) in enumerate(
                        vector_results, start=1
                    ):
                        chunk_id = document.metadata.get("chunk_id")

                        if chunk_id in self.records_by_id:
                            fused_scores[chunk_id] += 1 / (RRF_K + rank)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    self.vector_error = str(exc)
                    self.vector_store = None

            scores = self.bm25.get_scores(tokenize(variant))
            order = sorted(
                range(len(self.records)),
                key=lambda index: scores[index],
                reverse=True,
            )

            keyword_rank = 0

            for index in order:
                if scores[index] <= 0:
                    break

                keyword_rank += 1
                chunk_id = self.records[index]["chunk_id"]

                fused_scores[chunk_id] += 1 / (
                    RRF_K + keyword_rank
                )

                if keyword_rank >= KEYWORD_CANDIDATES:
                    break

        phrases = query_row_phrases(query)
        phrase_match_exists = any(
            table_has_query_phrase(record, phrases)
            for record in self.records
        )

        features = {}

        for record in self.records:
            chunk_id = record["chunk_id"]

            match = table_match_features(
                record,
                query,
                phrases=phrases,
                prefer_phrases=phrase_match_exists,
            )

            features[chunk_id] = match

            if match["row_match"]:
                fused_scores.setdefault(chunk_id, 0.0)

        candidate_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                fused_scores[chunk_id] + features[chunk_id]["bonus"]
            ),
            reverse=True,
        )[:max(top_k, RERANKER_CANDIDATES)]

        rerank_scores = {}
        if RERANKER_ENABLED and candidate_ids:
            try:
                reranker = self._get_reranker()
                pairs = [
                    (query, self.records_by_id[chunk_id]["text"])
                    for chunk_id in candidate_ids
                ]
                scores = reranker.predict(pairs)
                if len(scores) != len(candidate_ids):
                    raise ValueError(
                        "Cross-encoder returned a different number of "
                        "scores than candidate chunks."
                    )
                rerank_scores = {
                    chunk_id: float(score)
                    for chunk_id, score in zip(candidate_ids, scores)
                }
                ranked_ids = sorted(
                    candidate_ids,
                    key=lambda chunk_id: (
                        rerank_scores[chunk_id],
                        features[chunk_id]["bonus"],
                    ),
                    reverse=True,
                )[:top_k]
            except Exception as exc:
                self.reranker_error = str(exc)
                print(
                    "[Retrieval] Cross-encoder reranking failed; "
                    f"using hybrid ranking instead: {exc}"
                )
                ranked_ids = candidate_ids[:top_k]
        else:
            ranked_ids = candidate_ids[:top_k]

        return [
            {
                "record": self.records_by_id[chunk_id],
                "rrf_score": fused_scores[chunk_id],
                "final_score": rerank_scores.get(
                    chunk_id,
                    fused_scores[chunk_id] + features[chunk_id]["bonus"],
                ),
                "rerank_score": rerank_scores.get(chunk_id),
                "reranker_error": self.reranker_error,
                "row_match": features[chunk_id]["row_match"],
                "header_match": features[chunk_id]["header_match"],
            }
            for chunk_id in ranked_ids
        ]


def main():
    """Run a general retrieval query against a selected PDF."""
    parser = argparse.ArgumentParser(
        description="General hybrid retrieval for PDF documents."
    )

    parser.add_argument(
        "query",
        help="Question to search for",
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="ID of the PDF to search",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    retriever = HybridRetriever(args.document_id)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\n[Retrieval] Query: {args.query}")
    print(f"[Retrieval] Results: {len(results)}")

    for rank, result in enumerate(results, start=1):
        record = result["record"]

        print(
            f"\nRank: {rank}"
            f" | Score: {result['final_score']:.5f}"
            f" | Type: {record.get('content_type')}"
        )

        print(f"Pages: {record.get('page_numbers', [])}")
        print(f"Chunk ID: {record.get('chunk_id')}")
        print(f"Row match: {result['row_match']}")
        print(f"Header match: {result['header_match']}")
        print(record.get("text", "")[:900])


if __name__ == "__main__":
    main()