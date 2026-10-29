"""LangGraph workflow for routing, retrieval, and grounded answering."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .agent import (
    AnswerGenerator,
    build_query_plan,
    detect_intent,
    discover_documents,
    generate_conversation_response,
    resolve_followup_question,
)
from .retrieval import HybridRetriever


class WorkflowState(TypedDict, total=False):
    question: str
    chat_history: list[dict[str, str]]
    resolved_question: str
    intent: str
    document_ids: list[str]
    query_plan: dict[str, Any]
    sub_questions: list[str]
    retrieval_candidates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    citations: list[str]
    sources: list[dict[str, Any]]
    retry_count: int
    status: str
    answer: str
    errors: list[str]


def _deduplicate_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        record = item.get("record", {})
        key = record.get("chunk_id") or record.get("text", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def route_intent_node(state: WorkflowState) -> WorkflowState:
    question = state.get("resolved_question", state.get("question", ""))
    state["intent"] = detect_intent(question)
    state["status"] = "routed"
    return state


def analyze_question_node(state: WorkflowState) -> WorkflowState:
    question = state.get("resolved_question", state.get("question", ""))
    document_ids = state.get("document_ids") or discover_documents()
    plan = build_query_plan(question, document_ids)
    state["query_plan"] = {
        "intent": plan.intent,
        "document_scope": plan.document_scope,
        "rewritten_queries": plan.rewritten_queries,
        "sub_questions": plan.sub_questions,
        "needs_decomposition": plan.needs_decomposition,
    }
    state["sub_questions"] = plan.sub_questions or [question]
    state["document_ids"] = document_ids
    return state


def retrieval_node(state: WorkflowState) -> WorkflowState:
    question = state.get("resolved_question", state.get("question", ""))
    document_ids = state.get("document_ids") or discover_documents()
    if not document_ids:
        document_ids = ["doc_a7a0d8261a0923404fc45446"]

    evidence: list[dict[str, Any]] = []
    for doc_id in document_ids:
        try:
            retriever = HybridRetriever(doc_id)
            for query in state.get("sub_questions", [question]):
                if not query.strip():
                    continue
                results = retriever.search(query, top_k=5)
                evidence.extend(results)
        except Exception as exc:  # pragma: no cover - defensive path
            state.setdefault("errors", []).append(f"{doc_id}: {exc}")

    state["retrieval_candidates"] = evidence
    state["evidence"] = _deduplicate_evidence(evidence)
    return state


def evidence_assessment_node(state: WorkflowState) -> WorkflowState:
    evidence = state.get("evidence", [])
    if not evidence:
        retry_count = int(state.get("retry_count", 0)) + 1
        state["retry_count"] = retry_count
        if retry_count >= 2:
            state["status"] = "insufficient_evidence_final"
        else:
            state["status"] = "insufficient_evidence"
        return state

    state["status"] = "evidence_ready"
    citations = []
    seen = set()
    for item in evidence:
        record = item.get("record", {})
        doc_id = record.get("document_id", "unknown")
        pages = sorted({int(page) for page in record.get("page_numbers", []) if str(page).isdigit()})
        if pages:
            citation = f"{doc_id}:p{pages[0]}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
    state["citations"] = citations
    return state


def generate_answer_node(state: WorkflowState) -> WorkflowState:
    question = state.get("resolved_question", state.get("question", ""))
    evidence = state.get("evidence", [])
    document_ids = state.get("document_ids") or discover_documents()
    generator = AnswerGenerator()
    result = generator.generate(question, evidence, document_ids)
    state["answer"] = result["answer"]
    state["citations"] = result["citations"]
    state["sources"] = result["sources"]
    state["status"] = "answered"
    return state


def finalize_conversation_node(state: WorkflowState) -> WorkflowState:
    question = state.get("resolved_question", state.get("question", ""))
    state["answer"] = generate_conversation_response(
        question,
        state.get("chat_history", []),
    )
    state["citations"] = []
    state["sources"] = []
    state["status"] = "conversation_response"
    return state


def build_graph():
    """Construct the LangGraph workflow for routing and retrieval."""
    workflow = StateGraph(WorkflowState)
    workflow.add_node("route_intent", route_intent_node)
    workflow.add_node("resolve_followup", resolve_followup_node)
    workflow.add_node("analyze_question", analyze_question_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("assess", evidence_assessment_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("conversation", finalize_conversation_node)
    workflow.add_node("final", lambda state: state)

    workflow.set_entry_point("resolve_followup")
    workflow.add_edge("resolve_followup", "route_intent")
    workflow.add_conditional_edges(
        "route_intent",
        lambda state: state["intent"],
        {
            "conversation": "conversation",
            "knowledge": "analyze_question",
        },
    )
    workflow.add_edge("conversation", "final")
    workflow.add_edge("analyze_question", "retrieve")
    workflow.add_edge("retrieve", "assess")
    workflow.add_conditional_edges(
        "assess",
        lambda state: (
            "retry"
            if state.get("status") == "insufficient_evidence"
            and int(state.get("retry_count", 0)) < 2
            else "answer"
        ),
        {
            "retry": "retrieve",
            "answer": "generate",
        },
    )
    workflow.add_edge("generate", "final")
    workflow.set_finish_point("final")
    return workflow.compile()


def resolve_followup_node(state: WorkflowState) -> WorkflowState:
    """Create the current standalone query without carrying prior answers."""
    state["resolved_question"] = resolve_followup_question(
        state.get("question", ""),
        state.get("chat_history", []),
    )
    return state


def answer_question(
    question: str,
    document_ids: list[str] | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the workflow and return the final answer plus citations."""
    app = build_graph()
    result = app.invoke(
        {
            "question": question,
            "document_ids": document_ids or [],
            "chat_history": chat_history or [],
            "retry_count": 0,
        }
    )
    return {
        "intent": result.get("intent", detect_intent(question)),
        "resolved_question": result.get("resolved_question", question),
        "status": result.get("status", "unknown"),
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "sources": result.get("sources", []),
        "evidence": result.get("evidence", []),
        "errors": result.get("errors", []),
    }
