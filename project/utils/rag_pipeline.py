"""
RAG answer generation utilities for Smart Notes RAG Assistant.

This module is the final AI step in the RAG pipeline:
a user question and retrieved document chunks come in, a grounded answer comes out.

Why this file exists:
- Keep LLM setup separate from Streamlit UI code.
- Use the same retrieval path for chat answers and source display.
- Make the prompt easy for beginners to read, debug, and improve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .text_chunker import TextChunk
    from .vector_store import VectorSearchResult


DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RAGPipelineError(Exception):
    """Raised when a RAG answer cannot be generated safely."""


@dataclass
class ChatTurn:
    """One previous chat message used to preserve conversational context."""

    role: str
    content: str


@dataclass
class RAGResponse:
    """Answer returned by the RAG pipeline, along with the chunks used."""

    answer: str
    sources: list["VectorSearchResult"]
    context: str


def load_environment() -> None:
    """
    Load variables from .env when python-dotenv is installed.

    The app can still run with system environment variables if python-dotenv is
    unavailable, but requirements.txt includes it for normal setup.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env")


def get_llm_provider() -> str:
    """Return the selected LLM provider from environment variables."""
    load_environment()
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()

    if provider not in {"ollama", "openai"}:
        raise RAGPipelineError("LLM_PROVIDER must be either 'ollama' or 'openai'.")

    return provider


def create_ollama_llm() -> Any:
    """
    Create a LangChain chat model for local Ollama usage.

    Ollama should be running before this is called:
    ollama serve
    ollama pull mistral
    """
    load_environment()
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RAGPipelineError(
            "langchain-ollama is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return ChatOllama(model=model_name, base_url=base_url, temperature=0.2)


def create_openai_llm() -> Any:
    """
    Create a LangChain chat model for optional OpenAI usage.

    This path is optional. The default project experience uses local Ollama.
    """
    load_environment()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    if not api_key:
        raise RAGPipelineError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RAGPipelineError(
            "langchain-openai is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)


def create_llm() -> Any:
    """Create the configured LLM client."""
    provider = get_llm_provider()

    if provider == "openai":
        return create_openai_llm()

    return create_ollama_llm()


def validate_question(question: str) -> str:
    """Clean and validate the user question before retrieval or prompting."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise RAGPipelineError("Question cannot be empty.")

    return cleaned_question


def format_chat_history(chat_history: list[ChatTurn] | None) -> str:
    """
    Format prior messages for the prompt.

    We keep only simple role/content pairs so Streamlit session state can store
    chat history without depending on LangChain message objects.
    """
    if not chat_history:
        return "No previous conversation."

    formatted_turns: list[str] = []
    for turn in chat_history:
        role = turn.role.strip().title() or "User"
        content = turn.content.strip()
        if content:
            formatted_turns.append(f"{role}: {content}")

    return "\n".join(formatted_turns) if formatted_turns else "No previous conversation."


def build_rag_prompt(question: str, context: str, chat_history: list[ChatTurn] | None = None) -> str:
    """
    Build the prompt sent to the LLM.

    The instruction asks the model to answer from the provided context and admit
    when the documents do not contain enough information.
    """
    history_text = format_chat_history(chat_history)

    return f"""
You are Smart Notes RAG Assistant, a helpful AI that answers questions using uploaded documents.

Rules:
1. Answer using only the provided document context.
2. If the context does not contain the answer, say that the uploaded documents do not provide enough information.
3. Be concise, clear, and beginner-friendly.
4. When useful, mention the source chunk numbers that support the answer.

Document context:
{context}

Chat history:
{history_text}

User question:
{question}

Answer:
""".strip()


def extract_llm_text(response: Any) -> str:
    """
    Convert a LangChain response object into plain text.

    Different providers may return either a string or an object with a content
    attribute, so this helper keeps the rest of the app provider-agnostic.
    """
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()

    return str(response).strip()


def generate_llm_answer(prompt: str) -> str:
    """Send the completed RAG prompt to the configured LLM."""
    llm = create_llm()

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        provider = get_llm_provider()
        if provider == "ollama":
            raise RAGPipelineError(
                "Could not reach Ollama. Make sure Ollama is running and the selected model is pulled."
            ) from exc

        raise RAGPipelineError("Could not generate an answer from the configured LLM.") from exc

    answer = extract_llm_text(response)
    if not answer:
        raise RAGPipelineError("The LLM returned an empty answer.")

    return answer


def answer_question(
    question: str,
    index: Any,
    chunks: list["TextChunk"],
    chat_history: list[ChatTurn] | None = None,
    top_k: int | None = None,
) -> RAGResponse:
    """
    Run the full RAG flow for one user question.

    Flow:
    question -> semantic retrieval -> context block -> LLM prompt -> answer
    """
    cleaned_question = validate_question(question)

    try:
        from .retrieval import RetrievalError, results_to_context, semantic_search

        retrieved_results = semantic_search(
            query=cleaned_question,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )
        context = results_to_context(retrieved_results)
    except RetrievalError as exc:
        raise RAGPipelineError(str(exc)) from exc

    prompt = build_rag_prompt(
        question=cleaned_question,
        context=context,
        chat_history=chat_history,
    )
    answer = generate_llm_answer(prompt)

    return RAGResponse(answer=answer, sources=retrieved_results, context=context)
