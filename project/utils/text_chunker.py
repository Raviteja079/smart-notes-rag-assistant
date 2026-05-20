"""
Text chunking utilities for Smart Notes RAG Assistant.

This module is the second step in the RAG pipeline:
large extracted text comes in, smaller searchable chunks come out.

Why chunking matters:
- Embedding models work better with focused pieces of text.
- Retrieval is more accurate when each vector represents a specific idea.
- Overlap helps preserve context that sits near a chunk boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 180


class TextChunkingError(Exception):
    """Raised when text cannot be split into valid chunks."""


@dataclass
class TextChunk:
    """Container for one chunk of text and its position in the document."""

    text: str
    chunk_id: int
    start_index: int
    end_index: int
    source: str | None = None


def clean_text(text: str) -> str:
    """
    Normalize whitespace while keeping paragraph boundaries readable.

    We keep this step small on purpose. Heavy cleaning can accidentally remove
    useful context from resumes, reports, notes, or technical documents.
    """
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validate chunk size settings before splitting text.

    The overlap must be smaller than the chunk size so the chunking loop can
    always move forward.
    """
    if chunk_size <= 0:
        raise TextChunkingError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise TextChunkingError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise TextChunkingError("chunk_overlap must be smaller than chunk_size.")


def split_text_into_chunks(
    text: str,
    source: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks for embeddings and retrieval.

    Expected input:
    - text: extracted plain text from a PDF or TXT file
    - source: optional filename for citations and debugging

    Expected output:
    - a list of TextChunk objects with text and position metadata
    """
    validate_chunk_settings(chunk_size, chunk_overlap)

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise TextChunkingError("Cannot create chunks from empty text.")

    chunks: list[TextChunk] = []
    start_index = 0
    chunk_id = 0

    while start_index < len(cleaned_text):
        end_index = min(start_index + chunk_size, len(cleaned_text))
        chunk_text = cleaned_text[start_index:end_index].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_index=start_index,
                    end_index=end_index,
                    source=source,
                )
            )
            chunk_id += 1

        if end_index == len(cleaned_text):
            break

        start_index = end_index - chunk_overlap

    if not chunks:
        raise TextChunkingError("No valid chunks were created from the text.")

    return chunks


def chunks_to_text_list(chunks: list[TextChunk]) -> list[str]:
    """
    Convert TextChunk objects into plain strings.

    Embedding models usually only need the text content, while the app keeps
    the full TextChunk objects for source display and citations.
    """
    return [chunk.text for chunk in chunks]
