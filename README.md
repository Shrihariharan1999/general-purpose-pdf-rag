# Multimodal Document RAG

This project is a PDF-only retrieval-augmented generation system for document understanding, grounded question answering, and conversational assistant behavior. It uses Docling for PDF extraction, chunking and provenance tracking, sentence-transformer embeddings with Chroma, BM25 keyword retrieval, hybrid fusion, and LangGraph for orchestration.

## Architecture

- PDF ingestion uses Docling and saves page-level assets in `data/raw` and `data/processed`.
- Chunking keeps text, table, and image metadata separate and preserves page provenance.
- Source PDF filenames are stored with chunk/vector metadata and included in the answer evidence and citations.
- Each answer response also returns structured source references with PDF name, document ID, page numbers, chunk IDs, content type, section headings, available source-region provenance, and retrieved chunk text. The chat provides a dropdown to inspect each selected chunk.
- Embeddings are generated with `sentence-transformers/all-MiniLM-L6-v2` and stored in a persistent Chroma collection under `data/processed/vectorstore/chroma`.
- Retrieval combines semantic similarity and BM25 ranking with table-aware heuristics.
- A configurable Sentence Transformers cross-encoder reranks the hybrid candidates before returning the top results.
- Streamlit warms and caches the embedding and reranker models at startup; model weights are reused across chat reruns and are not reloaded when each question is asked.
- A LangGraph workflow routes greetings, personal conversation, and app-usage questions separately from factual or explanatory requests. Factual requests are checked against uploaded PDFs even if they do not mention a document or use a fixed keyword list.
- The Streamlit chat passes prior user questions to Gemini to resolve follow-up references. Prior assistant answers are excluded from contextualization, retrieval, and grounded generation.
- Grounded answers are generated with the configured LLM when API credentials are available. The LLM is instructed to answer only from relevant PDF evidence and explicitly abstain when the evidence is missing, unrelated, or contradictory. Without credentials, a conservative local fallback only summarizes evidence with matching terms and otherwise reports insufficient relevant information.

## Setup

1. Create and activate the project virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys as needed. `GOOGLE_API_KEY` and LangSmith tracing are optional.
4. Add PDF files to `data/raw` or upload them through the app.

## Ingest and index PDFs

Upload one or more PDFs in the Streamlit sidebar, then click **Process and index PDFs**. Each file is validated and checked by full SHA-256 content hash against prior manifests and Chroma vectors; already indexed exact duplicates are skipped. New files are processed through Docling extraction, type-aware chunking, Sentence Transformers embedding, and persistent Chroma indexing. Progress, completion details, duplicate skips, and per-file errors stay in the sidebar. The chat input is disabled while new uploads await processing. Uploaded files are limited to 100 MB each.

## Run the application

Start the app with:

`streamlit run streamlit_app.py`

You can then:

- Upload one or more PDFs
- Click **Process and index PDFs** and monitor progress in the sidebar
- Ask conversational or document-related questions
- View grounded answers with document/page citations

## Testing

Run the repository test suite with:

`pytest -q`

This validates routing behavior, graph execution, and document-query handling without requiring external API credentials.

## Notes and limitations

- The system expects PDF input only.
- Vision features stay limited to extracted page assets and table/picture crops; they are tracked in metadata but not always used as a full multimodal retrieval pipeline.
- External LLM services are optional; without credentials, the system still returns evidence-based fallback answers tied to indexed chunks.
- Cross-encoder weights are downloaded on first use unless already cached. Configure `RERANKER_MODEL`, `RERANKER_CANDIDATES`, or disable reranking with `RERANKER_ENABLED=false` in `.env`.
- Large PDFs or costly OCR settings can increase ingestion time substantially.
