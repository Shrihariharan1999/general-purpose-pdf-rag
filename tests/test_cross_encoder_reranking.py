from document_rag import retrieval
from document_rag import embeddings
from document_rag.retrieval import HybridRetriever


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


class FakeBM25:
    def get_scores(self, _query_tokens):
        return [2.0, 1.0]


def build_retriever(reranker):
    records = [
        {
            "document_id": "doc",
            "chunk_id": "chunk-1",
            "content_type": "text",
            "text": "evidence about alpha",
            "page_numbers": [1],
        },
        {
            "document_id": "doc",
            "chunk_id": "chunk-2",
            "content_type": "text",
            "text": "evidence about beta",
            "page_numbers": [2],
        },
    ]
    instance = object.__new__(HybridRetriever)
    instance.document_id = "doc"
    instance.records = records
    instance.records_by_id = {record["chunk_id"]: record for record in records}
    instance.bm25 = FakeBM25()
    instance.vector_store = None
    instance.reranker = reranker
    instance.reranker_error = None
    return instance


def test_cross_encoder_reranks_hybrid_candidates(monkeypatch):
    reranker = FakeCrossEncoder([0.1, 0.9])
    retriever = build_retriever(reranker)
    monkeypatch.setattr(retrieval, "RERANKER_ENABLED", True)
    monkeypatch.setattr(retrieval, "RERANKER_CANDIDATES", 2)

    results = retriever.search("evidence alpha beta", top_k=1)

    assert results[0]["record"]["chunk_id"] == "chunk-2"
    assert results[0]["rerank_score"] == 0.9
    assert reranker.pairs == [
        ("evidence alpha beta", "evidence about alpha"),
        ("evidence alpha beta", "evidence about beta"),
    ]


def test_reranker_failure_falls_back_to_hybrid_order(monkeypatch):
    class BrokenCrossEncoder:
        def predict(self, _pairs):
            raise RuntimeError("model unavailable")

    retriever = build_retriever(BrokenCrossEncoder())
    monkeypatch.setattr(retrieval, "RERANKER_ENABLED", True)
    monkeypatch.setattr(retrieval, "RERANKER_CANDIDATES", 2)

    results = retriever.search("evidence alpha beta", top_k=1)

    assert results[0]["record"]["chunk_id"] == "chunk-1"
    assert "model unavailable" in results[0]["reranker_error"]


def test_top_k_must_be_positive():
    retriever = build_retriever(FakeCrossEncoder([]))

    try:
        retriever.search("evidence", top_k=0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("Expected top_k validation to fail")


def test_embedding_model_is_loaded_once_and_reused(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    embeddings.create_embedding_model.cache_clear()
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", FakeEmbeddings)
    try:
        first = embeddings.create_embedding_model()
        second = embeddings.create_embedding_model()
    finally:
        embeddings.create_embedding_model.cache_clear()

    assert first is second
    assert len(calls) == 1


def test_reranker_model_is_loaded_once_and_reused(monkeypatch):
    import sentence_transformers

    calls = []
    model = object()

    def fake_cross_encoder(model_name):
        calls.append(model_name)
        return model

    retrieval._load_reranker.cache_clear()
    monkeypatch.setattr(
        sentence_transformers,
        "CrossEncoder",
        fake_cross_encoder,
    )
    try:
        first = retrieval._load_reranker("fake-reranker")
        second = retrieval._load_reranker("fake-reranker")
    finally:
        retrieval._load_reranker.cache_clear()

    assert first is second
    assert calls == ["fake-reranker"]
