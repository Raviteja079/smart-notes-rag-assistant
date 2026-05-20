"""
Streamlit UI helpers for Smart Notes RAG Assistant.

This module contains small display functions used by app.py.

Why this file exists:
- Keep app.py focused on user flow instead of repeated UI formatting.
- Make source chunks and chat messages display consistently.
- Keep Streamlit-specific code separate from the core RAG pipeline.
"""

from __future__ import annotations

from typing import Any


class UIHelperError(Exception):
    """Raised when UI helper input is invalid."""


def _import_streamlit() -> Any:
    """
    Import Streamlit only when a UI helper is called.

    Lazy imports let backend modules be syntax-checked and tested without
    requiring Streamlit to be imported immediately.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise UIHelperError("streamlit is not installed. Run: pip install -r requirements.txt") from exc

    return st


def render_app_header() -> None:
    """Render the main app title and short project description."""
    st = _import_streamlit()
    st.title("Smart Notes RAG Assistant")
    st.caption("Upload documents, search them semantically, and ask grounded questions.")


def render_upload_summary(uploaded_files: list[Any] | None) -> None:
    """
    Show a compact summary of files selected in the sidebar.

    Expected input:
    - Streamlit UploadedFile objects, or any objects with name and size fields
    """
    st = _import_streamlit()

    if not uploaded_files:
        st.info("No documents uploaded yet.")
        return

    st.subheader("Uploaded files")
    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "unknown file")
        size = getattr(uploaded_file, "size", None)
        size_label = format_file_size(size) if isinstance(size, int) else "unknown size"
        st.write(f"- {filename} ({size_label})")


def format_file_size(size_in_bytes: int) -> str:
    """Convert bytes into a readable KB/MB label for the UI."""
    if size_in_bytes < 0:
        raise UIHelperError("File size cannot be negative.")

    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"

    size_in_kb = size_in_bytes / 1024
    if size_in_kb < 1024:
        return f"{size_in_kb:.1f} KB"

    size_in_mb = size_in_kb / 1024
    return f"{size_in_mb:.1f} MB"


def show_success(message: str) -> None:
    """Display a success message in a consistent style."""
    st = _import_streamlit()
    st.success(message)


def show_warning(message: str) -> None:
    """Display a warning message in a consistent style."""
    st = _import_streamlit()
    st.warning(message)


def show_error(message: str) -> None:
    """Display an error message in a consistent style."""
    st = _import_streamlit()
    st.error(message)


def render_chat_history(chat_messages: list[dict[str, str]]) -> None:
    """
    Render chat history using Streamlit's chat message components.

    Expected input:
    - a list of dictionaries like {"role": "user", "content": "..."}
    """
    st = _import_streamlit()

    for message in chat_messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        if not content:
            continue

        with st.chat_message(role):
            st.markdown(content)


def render_source_chunks(results: list[Any], title: str = "Source chunks") -> None:
    """
    Display retrieved chunks used for semantic search or RAG answers.

    Expected input:
    - VectorSearchResult objects from vector_store.py
    """
    st = _import_streamlit()

    if not results:
        st.info("No source chunks to display.")
        return

    st.subheader(title)

    for result in results:
        chunk = result.chunk
        source = chunk.source or "unknown source"
        label = f"Rank {result.rank} | {source} | Chunk {chunk.chunk_id} | Score {result.score:.3f}"

        with st.expander(label):
            st.write(chunk.text)


def render_document_status(
    total_documents: int,
    total_chunks: int,
    vector_store_ready: bool,
) -> None:
    """
    Show the current document processing status.

    This gives users confidence that uploads were processed before they start
    asking questions.
    """
    st = _import_streamlit()

    status_text = "ready" if vector_store_ready else "not ready"

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", total_documents)
    col2.metric("Chunks", total_chunks)
    col3.metric("Vector store", status_text)


def render_semantic_search_results(results: list[Any]) -> None:
    """Display standalone semantic search results."""
    render_source_chunks(results, title="Semantic search results")


def render_processing_help() -> None:
    """Show a short hint when no vector store is available yet."""
    st = _import_streamlit()
    st.info("Upload PDF or TXT files from the sidebar, then click Process documents.")
