from __future__ import annotations

"""
Document loading utilities for Smart Notes RAG Assistant.

This module is the first step in the RAG pipeline:
uploaded files come in, readable text comes out.

Why this file exists:
- Keep PDF/TXT extraction separate from the Streamlit UI.
- Give later modules clean text instead of raw uploaded files.
- Convert low-level parsing failures into beginner-friendly errors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentLoadError(Exception):
    """Raised when an uploaded document cannot be read safely."""


@dataclass
class LoadedDocument:
    """Simple container for extracted document text and basic metadata."""

    filename: str
    text: str
    file_type: str


def get_file_extension(filename: str) -> str:
    """Return a lowercase file extension such as '.pdf' or '.txt'."""
    return Path(filename).suffix.lower()


def validate_supported_file(filename: str) -> str:
    """
    Check whether the uploaded file type is supported.

    We validate early so the rest of the pipeline only receives files it
    knows how to process.
    """
    extension = get_file_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(
            f"Unsupported file type '{extension or 'unknown'}'. Supported types: {allowed}."
        )

    return extension


def extract_text_from_pdf(file: BinaryIO) -> str:
    """
    Extract readable text from a PDF file.

    PDF parsing can fail for corrupted, encrypted, or scanned image-only PDFs.
    We raise DocumentLoadError so the UI can show a clean message instead of a
    long technical traceback.
    """
    try:
        reader = PdfReader(file)
    except Exception as exc:
        raise DocumentLoadError("Could not read this PDF. It may be corrupted or encrypted.") from exc

    page_texts: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise DocumentLoadError(f"Could not extract text from PDF page {page_number}.") from exc

        cleaned_page_text = page_text.strip()
        if cleaned_page_text:
            page_texts.append(cleaned_page_text)

    return "\n\n".join(page_texts).strip()


def extract_text_from_txt(file: BinaryIO) -> str:
    """
    Extract text from a TXT file using common encodings.

    Most modern text files are UTF-8, but a small fallback list makes the app
    friendlier for files saved by older editors or operating systems.
    """
    raw_bytes = file.read()

    if not raw_bytes:
        return ""

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue

    raise DocumentLoadError("Could not decode this text file.")


def load_uploaded_file(file: BinaryIO, filename: str) -> LoadedDocument:
    """
    Load an uploaded PDF or TXT file and return extracted text.

    This is the main function other parts of the app should call.
    This function is designed to work with Streamlit's UploadedFile object,
    but it also accepts any binary file-like object for easy testing.

    Pipeline role:
    uploaded file -> extracted plain text -> LoadedDocument
    """
    extension = validate_supported_file(filename)

    try:
        file.seek(0)
    except Exception:
        # Some file-like objects may not support seek; extraction can still work.
        pass

    if extension == ".pdf":
        text = extract_text_from_pdf(file)
    else:
        text = extract_text_from_txt(file)

    if not text:
        raise DocumentLoadError("No readable text was found in this document.")

    return LoadedDocument(filename=filename, text=text, file_type=extension.lstrip("."))
