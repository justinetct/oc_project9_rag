"""Recherche sémantique simple basée sur l'index FAISS local."""

from __future__ import annotations

from copy import deepcopy

from echo_app.indexing.embeddings import embed_query
from echo_app.indexing.faiss_store import load_vector_store, search_index


def format_search_result(
    chunk_metadata: dict,
    faiss_id: int,
    distance: float,
) -> dict:
    """Construit un résultat de recherche lisible à partir d'un chunk indexé."""
    return {
        "text": chunk_metadata.get("chunk_text"),
        "score": float(distance),
        "metadata": deepcopy(chunk_metadata.get("metadata", {})),
        "chunk_id": chunk_metadata.get("chunk_id"),
        "event_id": chunk_metadata.get("event_id"),
        "faiss_id": int(faiss_id),
        "distance": float(distance),
    }


def search_similar_events(query: str, top_k: int = 5) -> list[dict]:
    """Recherche les chunks d'événements les plus proches d'une requête texte."""
    if not query or not str(query).strip():
        raise ValueError("La requête de recherche est vide.")
    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")

    index, faiss_metadata = load_vector_store()
    query_embedding = embed_query(query)
    distances, indices = search_index(index, query_embedding, top_k=top_k)

    results: list[dict] = []
    for distance, faiss_id in zip(distances, indices, strict=True):
        chunk_metadata = faiss_metadata[faiss_id]
        results.append(format_search_result(chunk_metadata, faiss_id, distance))

    return results
