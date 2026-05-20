"""
Embedding utilities for Smart Notes RAG Assistant.

This module is the third step in the RAG pipeline:
text chunks come in, numeric vectors come out.

Why this file exists:
- Keep embedding model loading separate from Streamlit and FAISS.
- Reuse the same embedding model for documents and user queries.
- Normalize vectors so FAISS inner-product search behaves like cosine search.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated safely."""


def get_embedding_model_name() -> str:
    """
    Read the embedding model name from environment variables.

    The default model is small, fast, and strong enough for a beginner-friendly
    local RAG project.
    """
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str | None = None):
    """
    Load and cache the Sentence Transformer model.

    Loading the model can take a few seconds, so caching prevents the app from
    downloading or initializing it again on every Streamlit rerun.
    """
    selected_model = model_name or get_embedding_model_name()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        return SentenceTransformer(selected_model)
    except Exception as exc:
        raise EmbeddingError(f"Could not load embedding model: {selected_model}") from exc


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length for cosine similarity.

    FAISS can search normalized vectors with inner product. When both vectors
    have unit length, inner product equals cosine similarity.
    """
    if embeddings.size == 0:
        raise EmbeddingError("Cannot normalize empty embeddings.")

    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def validate_texts(texts: Iterable[str]) -> list[str]:
    """
    Remove empty strings and make sure we have text worth embedding.

    Empty chunks waste compute and create poor retrieval results, so we filter
    them out before calling the model.
    """
    cleaned_texts = [text.strip() for text in texts if text and text.strip()]

    if not cleaned_texts:
        raise EmbeddingError("No valid text was provided for embedding.")

    return cleaned_texts


def generate_text_embeddings(
    texts: Iterable[str],
    model_name: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate normalized embeddings for document chunks.

    Expected input:
    - an iterable of chunk strings

    Expected output:
    - a NumPy array shaped like: number_of_chunks x embedding_dimension
    """
    cleaned_texts = validate_texts(texts)
    model = load_embedding_model(model_name)

    try:
        embeddings = model.encode(
            cleaned_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError("Could not generate text embeddings.") from exc

    return normalize_embeddings(np.asarray(embeddings))


def generate_query_embedding(query: str, model_name: str | None = None) -> np.ndarray:
    """
    Generate one normalized embedding for a user search or chat question.

    The returned shape is 1 x embedding_dimension because FAISS expects a batch
    of query vectors, even when there is only one query.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise EmbeddingError("Query cannot be empty.")

    return generate_text_embeddings([cleaned_query], model_name=model_name)
