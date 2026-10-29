"""Minimal Streamlit app for ingestion and grounded document Q&A."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from document_rag.config import RAW_DATA_DIR
from document_rag.graph import answer_question
from document_rag.pipeline import find_indexed_pdf, process_pdf
from document_rag.retrieval import warm_retrieval_models

st.set_page_config(page_title="Multimodal PDF RAG", page_icon="📄")


@st.cache_resource(show_spinner="Loading local retrieval models...")
def load_retrieval_models():
    return warm_retrieval_models()


load_retrieval_models()

st.title("Multimodal PDF RAG")
st.caption("Upload PDFs, then process them from the sidebar.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = {}


with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

queued_uploads = []
if uploaded:
    for uploaded_file in uploaded:
        content = uploaded_file.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        status = st.session_state.upload_status.get(digest, {})
        if status.get("status") not in {"Indexed", "Skipped"}:
            original_name = Path(
                uploaded_file.name.replace("\\", "/")
            ).name
            previous = find_indexed_pdf(digest)
            if previous is not None:
                st.session_state.upload_status[digest] = {
                    "filename": original_name,
                    "status": "Skipped",
                    "document_id": previous["document_id"],
                    "page_count": previous["page_count"],
                    "chunk_count": previous["chunk_count"],
                }
            else:
                queued_uploads.append(
                    (uploaded_file, content, digest, original_name)
                )

with st.sidebar:
    if queued_uploads:
        st.caption(
            f"{len(queued_uploads)} PDF(s) ready to process. "
            "Chat is available after indexing."
        )
    process_clicked = st.button(
        "Process and index PDFs",
        disabled=not queued_uploads,
        use_container_width=True,
    )

    if process_clicked:
        extraction_converter = None
        embedding_model = None

        for uploaded_file, content, digest, original_name in queued_uploads:
            st.markdown(f"**{original_name}**")
            status_area = st.empty()
            progress_bar = st.progress(0, text="Queued")
            st.session_state.upload_status[digest] = {
                "filename": original_name,
                "status": "Processing",
            }

            try:
                if len(content) > 100 * 1024 * 1024:
                    raise ValueError("PDF exceeds the 100 MB upload limit.")
                if b"%PDF-" not in content[:1024]:
                    raise ValueError(
                        "The uploaded file does not have a valid PDF signature."
                    )

                previous = find_indexed_pdf(digest)
                if previous is not None:
                    st.session_state.upload_status[digest] = {
                        "filename": original_name,
                        "status": "Skipped",
                        "document_id": previous["document_id"],
                        "page_count": previous["page_count"],
                        "chunk_count": previous["chunk_count"],
                    }
                    progress_bar.progress(1.0, text="Already indexed")
                    status_area.info(
                        "Exact file content is already indexed; processing skipped."
                    )
                    continue

                RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
                target = RAW_DATA_DIR / f"{digest[:12]}_{original_name}"
                target.write_bytes(content)

                def report_progress(
                    message: str,
                    fraction: float | None,
                ) -> None:
                    status_area.caption(message)
                    if fraction is not None:
                        progress_bar.progress(
                            max(0.0, min(float(fraction), 1.0)),
                            text=message,
                        )

                if extraction_converter is None:
                    status_area.caption("Loading PDF processing models...")
                    from document_rag.ingestion import create_converter

                    extraction_converter = create_converter()
                if embedding_model is None:
                    status_area.caption("Loading embedding model...")
                    from document_rag.embeddings import create_embedding_model

                    embedding_model = create_embedding_model()

                result = process_pdf(
                    target,
                    converter=extraction_converter,
                    embedding_model=embedding_model,
                    on_progress=report_progress,
                )
                if result.get("already_indexed"):
                    st.session_state.upload_status[digest] = {
                        "filename": original_name,
                        "status": "Skipped",
                        "document_id": result["document_id"],
                        "page_count": result["page_count"],
                        "chunk_count": result["chunk_count"],
                    }
                    progress_bar.progress(1.0, text="Already indexed")
                    status_area.info(
                        "Exact file content is already indexed; processing skipped."
                    )
                    continue
                st.session_state.upload_status[digest] = {
                    "filename": original_name,
                    "status": "Indexed",
                    "document_id": result["document_id"],
                    "page_count": result["page_count"],
                    "chunk_count": result["chunk_count"],
                }
                progress_bar.progress(1.0, text="Indexed")
                status_area.success(
                    f"Indexed {result['page_count']} pages into "
                    f"{result['chunk_count']} chunks."
                )
            except Exception as exc:
                st.session_state.upload_status[digest] = {
                    "filename": original_name,
                    "status": "Failed",
                    "error": str(exc),
                }
                status_area.error(f"Processing failed: {exc}")

        queued_uploads = [
            item
            for item in queued_uploads
            if st.session_state.upload_status.get(item[2], {}).get("status")
            not in {"Indexed", "Skipped"}
        ]

    if st.session_state.upload_status:
        st.divider()
        st.subheader("Processing status")
        for item in st.session_state.upload_status.values():
            if item["status"] == "Indexed":
                st.success(
                    f"{item['filename']} — Indexed "
                    f"({item['page_count']} pages, "
                    f"{item['chunk_count']} chunks)"
                )
            elif item["status"] == "Skipped":
                st.info(
                    f"{item['filename']} — Already indexed; skipped "
                    f"({item['page_count']} pages, "
                    f"{item['chunk_count']} chunks)"
                )
            elif item["status"] == "Processing":
                st.info(f"{item['filename']} — Processing")
            elif item["status"] == "Failed":
                st.error(
                    f"{item['filename']} — "
                    f"{item.get('error', 'Failed')}"
                )

for message_index, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources and page references"):
                sources = message["sources"]

                def source_label(index: int) -> str:
                    source = sources[index]
                    pages = ", ".join(
                        str(page) for page in source.get("pages", [])
                    ) or "not recorded"
                    chunk_ids = ", ".join(source.get("chunk_ids", []))
                    return (
                        f"{source.get('document_name', 'Unknown PDF')} | "
                        f"p. {pages} | {chunk_ids}"
                    )

                selected_index = st.selectbox(
                    "Retrieved chunk",
                    options=range(len(sources)),
                    format_func=source_label,
                    key=f"source_chunk_{message_index}",
                )
                source = sources[selected_index]
                st.caption(
                    f"Document ID: {source.get('document_id', 'unknown')} | "
                    f"Type: {source.get('content_type', 'text')}"
                )
                if source.get("headings"):
                    st.caption("Section: " + " > ".join(source["headings"]))
                if source.get("source_regions"):
                    st.caption(
                        f"Source regions: {len(source['source_regions'])}"
                    )
                if source.get("rerank_score") is not None:
                    st.caption(
                        f"Cross-encoder score: "
                        f"{source['rerank_score']:.4f}"
                    )
                st.markdown("**Retrieved chunk content**")
                st.text(source.get("text", "Chunk content unavailable."))
        elif message.get("citations"):
            st.caption("Sources: " + ", ".join(message["citations"]))

question = st.chat_input(
    "Ask a question about your PDFs",
    disabled=any(
        st.session_state.upload_status.get(item[2], {}).get("status")
        in {None, "Processing"}
        for item in queued_uploads
    ),
)
if question:
    prior_history = list(st.session_state.chat_history)
    try:
        with st.spinner("Generating answer..."):
            result = answer_question(question, chat_history=prior_history)
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        sources = result.get("sources", [])
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "sources": sources,
            }
        )
    except Exception as exc:
        st.error(f"Could not answer the question: {exc}")
        st.stop()
    st.rerun()
