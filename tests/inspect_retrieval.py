"""Inspect top semantic retrieval results from the Chroma database."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "multimodal_pdf_chunks"
VECTOR_DB_DIR = Path("data/processed/vectorstore/chroma")
DOCUMENT_ID = "doc_a7a0d8261a0923404fc45446"

QUERIES = [
    "What was Apple's net income for the nine months ended June 25, 2022?",
    "Net income nine months ended June 25 2022",
]


def main():
    """Run test queries and inspect their highest-ranked source chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    print(f"Vectors stored: {store._collection.count()}")

    for query in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")

        results = store.similarity_search_with_score(
            query,
            k=10,
            filter={"document_id": DOCUMENT_ID},
        )

        target_found = False

        for rank, (document, distance) in enumerate(results, start=1):
            metadata = document.metadata
            pages = metadata.get("page_numbers", "")

            if isinstance(pages, str):
                page_list = pages.split(",")

            text = document.page_content
            is_target = (
                metadata.get("content_type") == "table"
                and "Net income" in text
                and "79,082" in text
                and "June 25, 2022" in text
                and "Nine Months Ended" in text
            )

            if is_target:
                target_found = True

            print(f"\nRank: {rank} | Distance: {distance:.4f}")
            print(f"Type: {metadata.get('content_type')}")
            print(f"Pages: {pages}")
            print(f"Chunk ID: {metadata.get('chunk_id')}")
            print(f"Expected table match: {is_target}")
            print(text[:900])

        if not target_found:
            print("\nExpected income-statement table not found in top 10.")


if __name__ == "__main__":
    main()