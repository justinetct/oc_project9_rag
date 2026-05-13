"""Utilitaire de recherche sémantique sur le vector store FAISS LangChain.

Ce module n'est plus utilisé par le service RAG en production (RagService
passe par ``retriever.invoke(question)``). Il reste exposé pour les
scripts manuels de démonstration (``scripts/06_test_semantic_search.py``)
en s'appuyant sur le même vector store ``langchain_community.vectorstores.FAISS``.
"""

from __future__ import annotations

from echo_app.indexing.langchain_embeddings import MistralLangChainEmbeddings
from echo_app.indexing.langchain_faiss_store import load_langchain_vector_store


_METADATA_TOP_LEVEL_KEYS = {"event_id", "chunk_id", "chunk_index", "chunk_count"}


def _format_search_result(document, distance: float) -> dict:
    """Reconstruit un dict compatible avec l'ancienne API à partir d'un Document."""
    metadata = document.metadata or {}
    nested_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in _METADATA_TOP_LEVEL_KEYS
    }
    return {
        "text": document.page_content,
        "score": float(distance),
        "metadata": nested_metadata,
        "chunk_id": metadata.get("chunk_id"),
        "event_id": metadata.get("event_id"),
        "distance": float(distance),
    }


def search_similar_events(query: str, top_k: int = 5) -> list[dict]:
    """Recherche les chunks d'événements les plus proches d'une requête texte."""
    if not query or not str(query).strip():
        raise ValueError("La requête de recherche est vide.")
    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")

    embeddings = MistralLangChainEmbeddings()
    vectorstore = load_langchain_vector_store(embeddings)
    hits = vectorstore.similarity_search_with_score(query, k=top_k)

    return [_format_search_result(document, distance) for document, distance in hits]
