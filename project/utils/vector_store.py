"""
FAISS vector store utilities for Smart Notes RAG Assistant.

This module is the fourth step in the RAG pipeline:
embeddings and text chunks come in, a searchable vector database comes out.

Why this file exists:
- Keep FAISS index creation separate from embedding generation.
- Store chunk metadata next to the vector index for source citations.
- Use normalized vectors with FAISS inner product to support cosine similarity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .text_chunker import TextChunk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_INDEX_FILENAME = "index.faiss"
CHUNK_METADATA_FILENAME = "chunks.json"


class VectorStoreError(Exception):
    """Raised when the FAISS vector store cannot be created, saved, or loaded."""


@dataclass
class VectorSearchResult:
    """One search result returned from the FAISS vector store."""

    chunk: TextChunk
    score: float
    rank: int


def _import_faiss() -> Any:
    """
    Import FAISS only when vector store functions need it.

    Lazy imports make the module easier to inspect and syntax-check before the
    full AI dependency stack is installed.
    """
    try:
        import faiss
    except ImportError as exc:
        raise VectorStoreError("faiss-cpu is not installed. Run: pip install -r requirements.txt") from exc

    return faiss


def _import_numpy() -> Any:
    """Import NumPy lazily for clearer setup errors."""
    try:
        import numpy as np
    except ImportError as exc:
        raise VectorStoreError("numpy is not installed. Run: pip install -r requirements.txt") from exc

    return np


def ensure_vectorstore_dir(vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR) -> Path:
    """Create the vector store folder if it does not already exist."""
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return vectorstore_dir


def validate_embeddings(embeddings: Any) -> Any:
    """
    Validate and convert embeddings before adding them to FAISS.

    FAISS expects a 2D float32 array shaped like:
    number_of_chunks x embedding_dimension
    """
    np = _import_numpy()
    embedding_array = np.asarray(embeddings, dtype="float32")

    if embedding_array.ndim != 2:
        raise VectorStoreError("Embeddings must be a 2D array.")

    if embedding_array.shape[0] == 0:
        raise VectorStoreError("Cannot build a vector store with zero embeddings.")

    if embedding_array.shape[1] == 0:
        raise VectorStoreError("Embedding dimension cannot be zero.")

    return embedding_array


def create_faiss_index(embeddings: Any) -> Any:
    """
    Create a FAISS index from normalized embeddings.

    We use IndexFlatIP because inner product on normalized vectors gives cosine
    similarity, which is what this RAG app needs for semantic search.
    """
    faiss = _import_faiss()
    embedding_array = validate_embeddings(embeddings)

    embedding_dimension = embedding_array.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embedding_array)

    return index


def save_chunk_metadata(
    chunks: list[TextChunk],
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
) -> Path:
    """
    Save chunk metadata as JSON.

    The FAISS index stores vectors only. This JSON file lets us map a search
    result back to the original chunk text, source filename, and character span.
    """
    ensure_vectorstore_dir(vectorstore_dir)
    metadata_path = vectorstore_dir / CHUNK_METADATA_FILENAME
    metadata = [asdict(chunk) for chunk in chunks]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_chunk_metadata(vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR) -> list[TextChunk]:
    """Load saved chunk metadata from JSON."""
    metadata_path = vectorstore_dir / CHUNK_METADATA_FILENAME

    if not metadata_path.exists():
        raise VectorStoreError(f"Chunk metadata file not found: {metadata_path}")

    try:
        raw_chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VectorStoreError("Chunk metadata file is not valid JSON.") from exc

    return [TextChunk(**chunk) for chunk in raw_chunks]


def save_faiss_index(index: Any, vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR) -> Path:
    """Save the FAISS index to disk."""
    faiss = _import_faiss()
    ensure_vectorstore_dir(vectorstore_dir)
    index_path = vectorstore_dir / FAISS_INDEX_FILENAME
    faiss.write_index(index, str(index_path))
    return index_path


def load_faiss_index(vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR) -> Any:
    """Load a previously saved FAISS index from disk."""
    faiss = _import_faiss()
    index_path = vectorstore_dir / FAISS_INDEX_FILENAME

    if not index_path.exists():
        raise VectorStoreError(f"FAISS index file not found: {index_path}")

    return faiss.read_index(str(index_path))


def build_and_save_vector_store(
    embeddings: Any,
    chunks: list[TextChunk],
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
) -> Any:
    """
    Build and persist a FAISS vector store.

    Expected input:
    - normalized embeddings from embedding_manager.py
    - TextChunk objects from text_chunker.py

    Expected output:
    - a FAISS index saved to disk
    - a chunks.json metadata file saved to disk
    """
    if len(chunks) == 0:
        raise VectorStoreError("Cannot build a vector store without chunks.")

    embedding_array = validate_embeddings(embeddings)

    if embedding_array.shape[0] != len(chunks):
        raise VectorStoreError("Number of embeddings must match number of chunks.")

    index = create_faiss_index(embedding_array)
    save_faiss_index(index, vectorstore_dir)
    save_chunk_metadata(chunks, vectorstore_dir)

    return index


def load_vector_store(vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR) -> tuple[Any, list[TextChunk]]:
    """
    Load the FAISS index and chunk metadata together.

    Retrieval needs both pieces: FAISS finds the nearest vector IDs, then the
    metadata list turns those IDs back into readable source chunks.
    """
    index = load_faiss_index(vectorstore_dir)
    chunks = load_chunk_metadata(vectorstore_dir)
    return index, chunks


def search_faiss_index(
    index: Any,
    query_embedding: Any,
    chunks: list[TextChunk],
    top_k: int = 4,
) -> list[VectorSearchResult]:
    """
    Search a FAISS index and return source chunks with similarity scores.

    This low-level function assumes query_embedding is already normalized.
    The retrieval module will provide a friendlier wrapper for app usage.
    """
    if top_k <= 0:
        raise VectorStoreError("top_k must be greater than 0.")

    if not chunks:
        raise VectorStoreError("Cannot search without chunk metadata.")

    np = _import_numpy()
    query_array = np.asarray(query_embedding, dtype="float32")

    if query_array.ndim == 1:
        query_array = query_array.reshape(1, -1)

    if query_array.ndim != 2 or query_array.shape[0] != 1:
        raise VectorStoreError("Query embedding must be a single vector or a 1 x dimension array.")

    search_limit = min(top_k, len(chunks))
    scores, indices = index.search(query_array, search_limit)

    results: list[VectorSearchResult] = []
    for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0]), start=1):
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        results.append(
            VectorSearchResult(
                chunk=chunks[int(chunk_index)],
                score=float(score),
                rank=rank,
            )
        )

    return results
