"""Chaîne RAG d'Écho : recherche sémantique et génération de réponses."""

from echo_app.rag.rag_service import (
    RagService,
    build_context,
    build_user_prompt,
    extract_sources,
)

__all__ = [
    "RagService",
    "build_context",
    "build_user_prompt",
    "extract_sources",
]
