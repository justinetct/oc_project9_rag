"""Outils d'indexation pour l'application Écho."""

from echo_app.indexing.embeddings import embed_query, embed_texts, get_mistral_client
from echo_app.indexing.faiss_store import (
    build_faiss_index,
    build_metadata,
    load_vector_store,
    save_vector_store,
    search_index,
)
from echo_app.indexing.search import search_similar_events

__all__ = [
    "embed_query",
    "embed_texts",
    "get_mistral_client",
    "build_faiss_index",
    "build_metadata",
    "load_vector_store",
    "save_vector_store",
    "search_index",
    "search_similar_events",
]
