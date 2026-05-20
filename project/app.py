"""
Smart Notes RAG Assistant - Streamlit application.

This file connects the full project together:
uploads -> text extraction -> chunking -> embeddings -> FAISS -> search/chat.

The heavy logic lives in the utils/ modules. app.py stays focused on user flow,
session state, and Streamlit layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from utils.embedding_manager import EmbeddingError, generate_text_embeddings
from utils.pdf_loader import DocumentLoadError, LoadedDocument, load_uploaded_file
from utils.rag_pipeline import ChatTurn, RAGPipelineError, answer_question
from utils.retrieval import RetrievalError, semantic_search
from utils.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunk,
    TextChunkingError,
    chunks_to_text_list,
    split_text_into_chunks,
)
from utils.ui_helpers import (
    render_app_header,
    render_chat_history,
    render_document_status,
    render_processing_help,
    render_semantic_search_results,
    render_source_chunks,
    render_upload_summary,
    show_error,
    show_success,
    show_warning,
)
from utils.vector_store import (
    DEFAULT_VECTORSTORE_DIR,
    VectorStoreError,
    build_and_save_vector_store,
    load_vector_store,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


load_dotenv(PROJECT_ROOT / ".env")


def get_env_int(name: str, default: int) -> int:
    """
    Read an integer from environment variables with a safe fallback.

    Streamlit sliders need valid integer defaults, so this helper keeps bad
    .env values from crashing the app.
    """
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Keep numeric UI defaults inside Streamlit slider bounds."""
    return max(minimum, min(value, maximum))


def initialize_session_state() -> None:
    """Create Streamlit session keys used across reruns."""
    defaults: dict[str, Any] = {
        "messages": [],
        "documents": [],
        "chunks": [],
        "index": None,
        "vector_store_ready": False,
        "latest_sources": [],
        "semantic_results": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_copy(uploaded_file: Any) -> Path:
    """
    Save a copy of the uploaded file into data/.

    This is useful for demos and debugging because the project keeps a local
    record of the documents used to build the vector store.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(uploaded_file.name).name
    file_path = DATA_DIR / safe_filename

    uploaded_file.seek(0)
    file_path.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)

    return file_path


def process_uploaded_documents(
    uploaded_files: list[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[LoadedDocument], list[TextChunk], Any]:
    """
    Convert uploaded files into a saved FAISS vector store.

    This function performs the core non-chat pipeline:
    load files -> chunk text -> embed chunks -> save FAISS index and metadata.
    """
    if not uploaded_files:
        raise DocumentLoadError("Please upload at least one PDF or TXT file.")

    documents: list[LoadedDocument] = []
    all_chunks: list[TextChunk] = []

    for uploaded_file in uploaded_files:
        save_uploaded_copy(uploaded_file)
        document = load_uploaded_file(uploaded_file, uploaded_file.name)
        documents.append(document)

        document_chunks = split_text_into_chunks(
            text=document.text,
            source=document.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Keep chunk IDs unique across multiple uploaded documents.
        for chunk in document_chunks:
            chunk.chunk_id = len(all_chunks) + chunk.chunk_id

        all_chunks.extend(document_chunks)

    chunk_texts = chunks_to_text_list(all_chunks)
    embeddings = generate_text_embeddings(chunk_texts)
    index = build_and_save_vector_store(embeddings, all_chunks, DEFAULT_VECTORSTORE_DIR)

    return documents, all_chunks, index


def load_existing_vector_store() -> None:
    """Load a vector store from disk if it already exists."""
    try:
        index, chunks = load_vector_store(DEFAULT_VECTORSTORE_DIR)
    except VectorStoreError as exc:
        show_warning(str(exc))
        return

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.vector_store_ready = True
    show_success("Loaded existing vector store.")


def clear_chat() -> None:
    """Reset chat-related state without deleting the vector store."""
    st.session_state.messages = []
    st.session_state.latest_sources = []


def clear_current_session() -> None:
    """Reset in-memory app state for a fresh session."""
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.vector_store_ready = False
    st.session_state.latest_sources = []
    st.session_state.semantic_results = []


def get_chat_history_for_prompt() -> list[ChatTurn]:
    """Convert Streamlit chat message dictionaries into RAG ChatTurn objects."""
    history: list[ChatTurn] = []

    for message in st.session_state.messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role and content:
            history.append(ChatTurn(role=role, content=content))

    return history


def render_sidebar() -> None:
    """Render uploads, model settings, and document processing controls."""
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        render_upload_summary(uploaded_files)

        st.header("Model settings")
        provider = st.selectbox(
            "LLM provider",
            options=["ollama", "openai"],
            index=0 if os.getenv("LLM_PROVIDER", "ollama").lower() == "ollama" else 1,
        )
        os.environ["LLM_PROVIDER"] = provider

        if provider == "ollama":
            ollama_model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "mistral"))
            ollama_base_url = st.text_input(
                "Ollama base URL",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
            os.environ["OLLAMA_MODEL"] = ollama_model
            os.environ["OLLAMA_BASE_URL"] = ollama_base_url
        else:
            openai_model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
            os.environ["OPENAI_MODEL"] = openai_model

        st.header("Chunking")
        default_chunk_size = clamp(get_env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE), 300, 2000)
        default_chunk_overlap = clamp(get_env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP), 0, 500)

        chunk_size = st.slider("Chunk size", min_value=300, max_value=2000, value=default_chunk_size, step=100)
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=default_chunk_overlap,
            step=20,
        )

        if st.button("Process documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents and building vector store..."):
                try:
                    documents, chunks, index = process_uploaded_documents(
                        uploaded_files=uploaded_files or [],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except (DocumentLoadError, TextChunkingError, EmbeddingError, VectorStoreError) as exc:
                    show_error(str(exc))
                else:
                    st.session_state.documents = documents
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.vector_store_ready = True
                    st.session_state.semantic_results = []
                    st.session_state.latest_sources = []
                    show_success(f"Processed {len(documents)} document(s) into {len(chunks)} chunks.")

        if st.button("Load existing vector store", use_container_width=True):
            load_existing_vector_store()

        if st.button("Clear chat", use_container_width=True):
            clear_chat()
            show_success("Chat history cleared.")

        if st.button("Reset session", use_container_width=True):
            clear_current_session()
            show_success("Session state reset.")


def render_semantic_search_section() -> None:
    """Render standalone semantic search controls and results."""
    st.subheader("Semantic Search")

    if not st.session_state.vector_store_ready:
        render_processing_help()
        return

    search_query = st.text_input("Search your documents", placeholder="Example: What are the main conclusions?")

    if st.button("Run semantic search", use_container_width=True):
        with st.spinner("Searching for relevant chunks..."):
            try:
                results = semantic_search(
                    query=search_query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                )
            except RetrievalError as exc:
                show_error(str(exc))
                st.session_state.semantic_results = []
            else:
                st.session_state.semantic_results = results
                show_success(f"Found {len(results)} relevant chunk(s).")

    render_semantic_search_results(st.session_state.semantic_results)


def render_chat_section() -> None:
    """Render the conversational RAG chat interface."""
    st.subheader("Chat With Your Notes")

    if not st.session_state.vector_store_ready:
        render_processing_help()
        return

    render_chat_history(st.session_state.messages)

    user_question = st.chat_input("Ask a question about your uploaded documents")
    if not user_question:
        return

    chat_history = get_chat_history_for_prompt()
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                response = answer_question(
                    question=user_question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    chat_history=chat_history,
                )
            except RAGPipelineError as exc:
                error_message = str(exc)
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.session_state.latest_sources = []
            else:
                st.markdown(response.answer)
                st.session_state.messages.append({"role": "assistant", "content": response.answer})
                st.session_state.latest_sources = response.sources


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Smart Notes RAG Assistant",
        layout="wide",
    )
    initialize_session_state()

    render_sidebar()
    render_app_header()

    render_document_status(
        total_documents=len(st.session_state.documents),
        total_chunks=len(st.session_state.chunks),
        vector_store_ready=st.session_state.vector_store_ready,
    )

    chat_tab, search_tab, sources_tab = st.tabs(["Chat", "Semantic Search", "Latest Sources"])

    with chat_tab:
        render_chat_section()

    with search_tab:
        render_semantic_search_section()

    with sources_tab:
        render_source_chunks(st.session_state.latest_sources, title="Sources used for latest answer")


if __name__ == "__main__":
    main()
