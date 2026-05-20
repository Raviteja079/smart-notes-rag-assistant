"""
Semantic retrieval utilities for Smart Notes RAG Assistant.

This module is the fifth step in the RAG pipeline:
a user query comes in, the most relevant document chunks come out.

Why this file exists:
- Hide FAISS and embedding details behind a simple semantic_search function.
- Reuse the same retrieval logic for standalone search and RAG chat.
- Return source chunks with scores so users can inspect why an answer was made.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .embedding_manager import EmbeddingError, generate_query_embedding
from .text_chunker import TextChunk
from .vector_store import (
    DEFAULT_VECTORSTORE_DIR,
    VectorSearchResult,
    VectorStoreError,
    load_vector_store,
    search_faiss_index,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_TOP_K = 4


class RetrievalError(Exception):
    """Raised when semantic retrieval cannot return useful results."""


def get_default_top_k() -> int:
    """
    Read the default number of retrieval results from environment variables.

    We keep a safe fallback so the app still works if the .env file is missing
    or contains a non-numeric value.
    """
    raw_top_k = os.getenv("TOP_K_RESULTS", str(DEFAULT_TOP_K))

    try:
        top_k = int(raw_top_k)
    except ValueError:
        return DEFAULT_TOP_K

    return top_k if top_k > 0 else DEFAULT_TOP_K


def validate_query(query: str) -> str:
    """Clean and validate a user search query before embedding it."""
    cleaned_query = query.strip()

    if not cleaned_query:
        raise RetrievalError("Search query cannot be empty.")

    return cleaned_query


def semantic_search(
    query: str,
    index: Any,
    chunks: list[TextChunk],
    top_k: int | None = None,
) -> list[VectorSearchResult]:
    """
    Retrieve the most relevant chunks for a user query.

    Expected input:
    - query: a natural-language question or search phrase
    - index: loaded FAISS index
    - chunks: metadata list matching the FAISS index order

    Expected output:
    - ranked VectorSearchResult objects containing chunk text and score
    """
    cleaned_query = validate_query(query)
    selected_top_k = top_k if top_k is not None else get_default_top_k()

    try:
        query_embedding = generate_query_embedding(cleaned_query)
        results = search_faiss_index(
            index=index,
            query_embedding=query_embedding,
            chunks=chunks,
            top_k=selected_top_k,
        )
    except EmbeddingError as exc:
        raise RetrievalError(str(exc)) from exc
    except VectorStoreError as exc:
        raise RetrievalError(str(exc)) from exc

    if not results:
        raise RetrievalError("No relevant document chunks were found.")

    return results


def semantic_search_from_saved_store(
    query: str,
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
    top_k: int | None = None,
) -> list[VectorSearchResult]:
    """
    Load the saved vector store and run semantic search.

    This helper is convenient for quick tests and for app flows where the
    vector store has already been persisted to disk.
    """
    try:
        index, chunks = load_vector_store(vectorstore_dir)
    except VectorStoreError as exc:
        raise RetrievalError(
            "Vector store not found. Please process documents before searching."
        ) from exc

    return semantic_search(query=query, index=index, chunks=chunks, top_k=top_k)


def results_to_context(results: list[VectorSearchResult]) -> str:
    """
    Convert retrieval results into a readable context block for RAG prompts.

    The RAG pipeline will inject this text into the LLM prompt so answers stay
    grounded in the uploaded documents.
    """
    if not results:
        raise RetrievalError("Cannot build context from empty retrieval results.")

    context_parts: list[str] = []
    for result in results:
        source = result.chunk.source or "unknown source"
        context_parts.append(
            f"[Source: {source}, Chunk: {result.chunk.chunk_id}, Score: {result.score:.3f}]\n"
            f"{result.chunk.text}"
        )

    return "\n\n---\n\n".join(context_parts)
