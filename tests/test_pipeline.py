from pathlib import Path
import json

import pytest

from document_rag import pipeline


def test_find_indexed_pdf_requires_matching_manifest_chunks_and_vectors(
    tmp_path,
    monkeypatch,
):
    import chromadb

    doc_id = "doc_existing"
    content_hash = "a" * 64
    manifest_dir = tmp_path / "docling" / doc_id
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "document_id": doc_id,
                "content_sha256": content_hash,
                "source_file": "report.pdf",
                "page_count": 8,
            }
        ),
        encoding="utf-8",
    )
    chunks_dir = tmp_path / "chunks" / f"{doc_id}_typeaware"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text(
        '{"chunk_id":"chunk-1"}\n{"chunk_id":"chunk-2"}\n',
        encoding="utf-8",
    )
    vector_dir = tmp_path / "vectorstore"
    vector_dir.mkdir()
    calls = []

    class FakeCollection:
        def get(self, where, include):
            calls.append(where)
            assert include == []
            return {"ids": ["chunk-1", "chunk-2"]}

    class FakeClient:
        def get_collection(self, name):
            assert name == pipeline.COLLECTION_NAME
            return FakeCollection()

    monkeypatch.setattr(pipeline, "DOCLING_OUTPUT_DIR", tmp_path / "docling")
    monkeypatch.setattr(pipeline, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(pipeline, "VECTOR_DB_DIR", vector_dir)
    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda path: FakeClient(),
    )

    result = pipeline.find_indexed_pdf(content_hash)

    assert result is not None
    assert result["document_id"] == doc_id
    assert result["source_file"] == "report.pdf"
    assert result["page_count"] == 8
    assert result["chunk_count"] == 2
    assert result["vector_count"] == 2
    assert calls == [{"document_id": doc_id}]
    assert pipeline.find_indexed_pdf("b" * 64) is None


def test_find_indexed_pdf_does_not_skip_when_vectors_are_missing(
    tmp_path,
    monkeypatch,
):
    import chromadb

    doc_id = "doc_incomplete"
    content_hash = "c" * 64
    manifest_dir = tmp_path / "docling" / doc_id
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "document_id": doc_id,
                "content_sha256": content_hash,
            }
        ),
        encoding="utf-8",
    )
    chunks_dir = tmp_path / "chunks" / f"{doc_id}_typeaware"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text(
        '{"chunk_id":"chunk-1"}\n',
        encoding="utf-8",
    )
    vector_dir = tmp_path / "vectorstore"
    vector_dir.mkdir()

    class FakeCollection:
        def get(self, where, include):
            assert include == []
            return {"ids": []}

    class FakeClient:
        def get_collection(self, name):
            return FakeCollection()

    monkeypatch.setattr(pipeline, "DOCLING_OUTPUT_DIR", tmp_path / "docling")
    monkeypatch.setattr(pipeline, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(pipeline, "VECTOR_DB_DIR", vector_dir)
    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda path: FakeClient(),
    )

    assert pipeline.find_indexed_pdf(content_hash) is None


def test_process_pdf_skips_hash_already_indexed(tmp_path, monkeypatch):
    pdf_path = tmp_path / "duplicate.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nsynthetic PDF")
    existing = {
        "document_id": "doc_existing",
        "source_file": "old-name.pdf",
        "page_count": 3,
        "chunk_count": 5,
        "vector_count": 5,
        "manifest": {},
    }
    monkeypatch.setattr(
        pipeline,
        "find_indexed_pdf",
        lambda _content_hash: existing,
    )

    def should_not_ingest(*_args, **_kwargs):
        raise AssertionError("Duplicate PDFs must not be reprocessed")

    monkeypatch.setattr(pipeline, "ingest_pdf", should_not_ingest)
    updates = []
    result = pipeline.process_pdf(
        pdf_path,
        on_progress=lambda message, progress: updates.append((message, progress)),
    )

    assert result["already_indexed"] is True
    assert result["document_id"] == "doc_existing"
    assert updates == [("Already indexed; skipped duplicate PDF", 1.0)]


def test_process_pdf_runs_extraction_chunking_and_indexing(tmp_path, monkeypatch):
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nsynthetic PDF")
    calls = []

    def fake_ingest(path, converter=None, on_progress=None):
        calls.append(("ingest", path.name, converter))
        return {"document_id": "doc_test", "page_count": 2}

    def fake_chunk(document_id, on_progress=None):
        calls.append(("chunk", document_id))
        return {"total_chunks": 4}

    def fake_index(document_id, embedding_model=None, on_progress=None):
        calls.append(("index", document_id, embedding_model))

    monkeypatch.setattr(pipeline, "ingest_pdf", fake_ingest)
    monkeypatch.setattr(pipeline, "chunk_document", fake_chunk)
    monkeypatch.setattr(pipeline, "index_document", fake_index)

    updates = []
    result = pipeline.process_pdf(
        pdf_path,
        converter="converter",
        embedding_model="embedding-model",
        on_progress=lambda message, progress: updates.append((message, progress)),
    )

    assert calls == [
        ("ingest", "example.pdf", "converter"),
        ("chunk", "doc_test"),
        ("index", "doc_test", "embedding-model"),
    ]
    assert result["document_id"] == "doc_test"
    assert result["page_count"] == 2
    assert result["chunk_count"] == 4
    assert updates[-1] == ("Processing complete", 1.0)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("not-a-pdf.txt", b"%PDF-1.7", "Expected a PDF"),
        ("empty.pdf", b"", "empty"),
        ("fake.pdf", b"not a PDF", "signature"),
    ],
)
def test_validate_pdf_rejects_invalid_files(tmp_path, filename, content, message):
    path = tmp_path / filename
    path.write_bytes(content)

    with pytest.raises(ValueError, match=message):
        pipeline.validate_pdf(path)


def test_validate_pdf_rejects_files_over_limit(tmp_path, monkeypatch):
    path = tmp_path / "large.pdf"
    path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(pipeline, "MAX_PDF_BYTES", 1)

    with pytest.raises(ValueError, match="upload limit"):
        pipeline.validate_pdf(path)
