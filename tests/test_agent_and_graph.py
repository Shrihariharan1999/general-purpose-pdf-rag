from document_rag import agent
from document_rag.agent import (
    AnswerGenerator,
    build_query_plan,
    detect_intent,
    resolve_followup_question,
)
from document_rag.embeddings import build_metadata
from document_rag.graph import answer_question


def test_general_conversation_is_not_routed_to_rag():
    assert detect_intent("Hi there!") == "conversation"
    assert detect_intent("What can this application do?") == "conversation"


def test_personal_introductions_stay_conversational_and_factual_questions_use_rag(
    monkeypatch,
):
    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "")

    assert detect_intent("I am shri") == "conversation"
    assert detect_intent("What is my name?") == "conversation"
    assert detect_intent("Explain photosynthesis") == "knowledge"
    assert detect_intent("What is quantum gravity?") == "knowledge"


def test_mixed_greeting_and_knowledge_question_routes_to_rag():
    assert detect_intent("Hi, can you summarize this PDF?") == "knowledge"


def test_build_query_plan_creates_sub_questions_for_complex_requests():
    plan = build_query_plan("Compare both approaches and explain their limitations.")
    assert plan.intent == "knowledge"
    assert len(plan.sub_questions) >= 2


def test_answer_question_handles_general_conversation_without_error():
    from document_rag import graph

    graph_generate = graph.generate_conversation_response
    graph.generate_conversation_response = lambda question, _history: (
        "Hi! How can I help you today?"
    )
    try:
        result = answer_question("Hi there!")
    finally:
        graph.generate_conversation_response = graph_generate

    assert result["status"] == "conversation_response"
    assert "help" in result["answer"].lower()


def test_conversation_fallback_remembers_user_name(monkeypatch):
    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "")

    introduction = answer_question("I am shri")
    answer = answer_question(
        "What is my name?",
        chat_history=[
            {"role": "user", "content": "I am shri"},
            {"role": "assistant", "content": introduction["answer"]},
        ],
    )

    assert introduction["answer"] == "Nice to meet you, Shri!"
    assert answer["intent"] == "conversation"
    assert answer["answer"] == "You told me your name is shri."


def test_conversation_model_receives_chat_history(monkeypatch):
    class FakeResponse:
        content = "You told me your name is Shri."

    captured_messages = []

    class FakeChatModel:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, messages):
            captured_messages.extend(messages)
            return FakeResponse()

    import langchain_google_genai

    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeChatModel,
    )

    answer = agent.generate_conversation_response(
        "What is my name?",
        [
            {"role": "user", "content": "I am Shri"},
            {"role": "assistant", "content": "Nice to meet you."},
        ],
    )

    assert answer == "You told me your name is Shri."
    assert ("user", "I am Shri") in captured_messages
    assert ("assistant", "Nice to meet you.") in captured_messages


def test_answer_question_handles_knowledge_request_without_crashing():
    result = answer_question("Summarize the main findings in the uploaded document.")
    assert result["status"] in {"answered", "conversation_response", "insufficient_evidence_final"}


def test_followup_resolution_uses_user_questions_but_not_assistant_answers(monkeypatch):
    class FakeResponse:
        content = "What does the report say about revenue?"

    captured_messages = []

    class FakeChatModel:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, messages):
            captured_messages.extend(messages)
            return FakeResponse()

    import langchain_google_genai

    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeChatModel,
    )

    resolved = resolve_followup_question(
        "What about revenue?",
        [
            {"role": "user", "content": "Summarize the report."},
            {"role": "assistant", "content": "It reports several metrics."},
        ],
    )

    assert resolved == "What does the report say about revenue?"
    prompt_content = str(captured_messages)
    assert "Summarize the report." in prompt_content
    assert "It reports several metrics." not in prompt_content
    assert "What about revenue?" in prompt_content


def test_followup_without_model_keeps_current_question(monkeypatch):
    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "")
    question = "What about the second one?"

    assert resolve_followup_question(
        question,
        [{"role": "user", "content": "Compare the two options."}],
    ) == question


def test_source_document_name_is_in_vector_metadata_and_llm_context():
    record = {
        "document_id": "doc_abc123",
        "document_name": "research-paper.pdf",
        "chunk_id": "chunk-1",
        "text": "The study reports a result.",
        "page_numbers": [4],
    }

    metadata = build_metadata(record)
    evidence = [{"record": record}]
    context = AnswerGenerator()._evidence_context(evidence)
    citations = AnswerGenerator()._citations_from_evidence(evidence)
    sources = AnswerGenerator()._sources_from_evidence(evidence)

    assert metadata["document_name"] == "research-paper.pdf"
    assert "Document name: research-paper.pdf" in context
    assert "research-paper.pdf (p. 4; ID: doc_abc123)" in citations
    assert sources == [
        {
            "document_name": "research-paper.pdf",
            "document_id": "doc_abc123",
            "pages": [4],
            "chunk_ids": ["chunk-1"],
            "content_type": "text",
            "headings": [],
            "source_regions": [],
            "text": "The study reports a result.",
            "retrieval_score": None,
            "rerank_score": None,
        }
    ]


def test_sources_preserve_multiple_pages_and_source_regions():
    record = {
        "document_id": "doc_abc123",
        "document_name": "research-paper.pdf",
        "chunk_id": "chunk-2",
        "content_type": "table",
        "text": "A table",
        "page_numbers": [4, 5],
        "headings": ["Results"],
        "source_regions": [
            {"page_number": 4, "element_type": "table"},
            {"page_number": 5, "element_type": "table"},
        ],
    }

    source = AnswerGenerator()._sources_from_evidence(
        [{"record": record, "final_score": 0.8, "rerank_score": 0.6}]
    )[0]

    assert source["pages"] == [4, 5]
    assert source["chunk_ids"] == ["chunk-2"]
    assert source["headings"] == ["Results"]
    assert len(source["source_regions"]) == 2
    assert source["text"] == "A table"
    assert source["rerank_score"] == 0.6


def test_answer_generator_extracts_text_from_gemini_content_blocks(monkeypatch):
    class FakeResponse:
        content = [
            {
                "type": "text",
                "text": (
                    "Revenue was **`15,338 million** "
                    "(approximately `15.34 billion) [report.pdf, p. 4]."
                ),
                "extras": {"signature": "internal-signature"},
            }
        ]

    class FakeChatModel:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _messages):
            return FakeResponse()

    import langchain_google_genai

    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeChatModel,
    )
    evidence = [
        {
            "record": {
                "document_id": "doc_1",
                "document_name": "report.pdf",
                "chunk_id": "chunk-1",
                "content_type": "text",
                "text": "Total revenue was $15.3 billion.",
                "page_numbers": [4],
            }
        }
    ]

    result = AnswerGenerator().generate("What was revenue?", evidence)

    assert result["answer"] == (
        "Revenue was **15,338 million** "
        "(approximately 15.34 billion) [report.pdf, p. 4]."
    )
    assert "signature" not in result["answer"]


def test_answer_generator_reports_gemini_call_errors(monkeypatch):
    class BrokenChatModel:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _messages):
            raise RuntimeError("model unavailable")

    import langchain_google_genai

    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        BrokenChatModel,
    )
    evidence = [{"record": {"document_id": "doc_1", "text": "evidence"}}]

    import pytest

    with pytest.raises(RuntimeError, match="model unavailable"):
        AnswerGenerator().generate("Question?", evidence)


def test_answer_generator_abstains_when_fallback_evidence_is_unrelated(monkeypatch):
    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "")
    evidence = [
        {
            "record": {
                "document_id": "doc_1",
                "document_name": "report.pdf",
                "chunk_id": "chunk-1",
                "text": "Revenue increased by 12% during the quarter.",
                "page_numbers": [2],
            }
        }
    ]

    result = AnswerGenerator().generate("What is quantum gravity?", evidence)

    assert "not find enough relevant information" in result["answer"]
    assert result["citations"] == []
    assert result["sources"] == []


def test_intent_classifier_routes_specific_information_to_rag(monkeypatch):
    class FakeResponse:
        content = "knowledge"

    captured_messages = []

    class FakeChatModel:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, messages):
            captured_messages.extend(messages)
            return FakeResponse()

    import langchain_google_genai

    monkeypatch.setattr(agent, "GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeChatModel,
    )

    assert detect_intent("Explain photosynthesis") == "knowledge"
    classifier_prompt = str(captured_messages)
    assert "every request for factual" in classifier_prompt
    assert "instead of guessing" in classifier_prompt
