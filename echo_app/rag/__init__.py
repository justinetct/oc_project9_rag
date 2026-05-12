"""Chaîne RAG d'Écho : recherche sémantique et génération de réponses."""

from echo_app.rag.langchain_chain import build_langchain_messages
from echo_app.rag.prompts import RAG_SYSTEM_PROMPT, build_user_prompt
from echo_app.rag.rag_service import (
    RagService,
    build_context,
    extract_sources,
)

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RagService",
    "build_context",
    "build_langchain_messages",
    "build_user_prompt",
    "extract_sources",
]
