"""Tests unitaires pour la recherche sémantique FAISS."""

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.indexing import search  # noqa: E402


def make_faiss_metadata() -> list[dict]:
    """Construit un mapping de métadonnées factice pour les tests."""
    return [
        {
            "faiss_id": 0,
            "chunk_id": "evt-1_0",
            "event_id": "evt-1",
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_text": "Soirée astronomie à Lanton.",
            "metadata": {
                "title": "Initiation à l'astronomie",
                "city": "Lanton",
                "url": "https://example.com/evt-1",
                "start_date": "2026-01-05T19:30",
            },
        },
        {
            "faiss_id": 1,
            "chunk_id": "evt-2_0",
            "event_id": "evt-2",
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_text": "Concert acoustique à Andernos.",
            "metadata": {
                "title": "Concert acoustique",
                "city": "Andernos-les-Bains",
                "url": "https://example.com/evt-2",
                "start_date": "2026-06-10T20:00",
            },
        },
        {
            "faiss_id": 2,
            "chunk_id": "evt-3_0",
            "event_id": "evt-3",
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_text": "Exposition au Bassin d'Arcachon.",
            "metadata": {
                "title": "Exposition",
                "city": "Arcachon",
                "url": "https://example.com/evt-3",
            },
        },
    ]


def install_fake_vector_store(monkeypatch, top_k_distances=None, top_k_indices=None) -> dict:
    """Installe des mocks pour load_vector_store, embed_query et search_index."""
    fake_index = object()
    fake_metadata = make_faiss_metadata()
    calls: dict = {}

    def fake_load_vector_store():
        calls["load_called"] = True
        return fake_index, fake_metadata

    def fake_embed_query(query: str) -> list[float]:
        calls["embed_query"] = query
        return [0.1, 0.2, 0.3]

    def fake_search_index(index, query_embedding, top_k: int = 5):
        calls["search_top_k"] = top_k
        default_distances = [0.10, 0.25, 0.40]
        default_indices = [0, 1, 2]
        distances = (top_k_distances or default_distances)[:top_k]
        indices = (top_k_indices or default_indices)[:top_k]
        return distances, indices

    monkeypatch.setattr(search, "load_vector_store", fake_load_vector_store)
    monkeypatch.setattr(search, "embed_query", fake_embed_query)
    monkeypatch.setattr(search, "search_index", fake_search_index)

    return calls


def test_search_similar_events_returns_expected_metadata(monkeypatch) -> None:
    """Le premier résultat doit contenir les métadonnées du chunk attendu."""
    install_fake_vector_store(monkeypatch)

    results = search.search_similar_events("astronomie", top_k=2)

    assert len(results) == 2
    first = results[0]
    assert first["chunk_id"] == "evt-1_0"
    assert first["event_id"] == "evt-1"
    assert first["metadata"]["title"] == "Initiation à l'astronomie"
    assert first["metadata"]["city"] == "Lanton"


def test_search_similar_events_respects_top_k(monkeypatch) -> None:
    """La taille des résultats et l'appel à search_index doivent suivre top_k."""
    calls = install_fake_vector_store(monkeypatch)

    results = search.search_similar_events("astronomie", top_k=1)

    assert len(results) == 1
    assert calls["search_top_k"] == 1


def test_search_similar_events_raises_when_query_is_empty(monkeypatch) -> None:
    """Une requête vide ou blanche doit être rejetée clairement."""
    install_fake_vector_store(monkeypatch)

    with pytest.raises(ValueError, match="requête de recherche est vide"):
        search.search_similar_events("   ", top_k=3)


def test_search_similar_events_raises_when_top_k_is_invalid(monkeypatch) -> None:
    """top_k doit rester strictement positif."""
    install_fake_vector_store(monkeypatch)

    with pytest.raises(ValueError, match="strictement positif"):
        search.search_similar_events("astronomie", top_k=0)


def test_search_similar_events_result_contains_all_expected_keys(monkeypatch) -> None:
    """Chaque résultat doit exposer text, score, metadata et les identifiants."""
    install_fake_vector_store(monkeypatch)

    results = search.search_similar_events("astronomie", top_k=1)

    first = results[0]
    assert set(first.keys()) == {
        "text",
        "score",
        "metadata",
        "chunk_id",
        "event_id",
        "faiss_id",
        "distance",
    }
    assert first["text"] == "Soirée astronomie à Lanton."
    assert first["score"] == pytest.approx(0.10)
    assert first["distance"] == pytest.approx(0.10)
    assert first["faiss_id"] == 0


def test_search_similar_events_propagates_missing_vector_store(monkeypatch) -> None:
    """Si le vector store est absent, l'erreur doit remonter au caller."""

    def fake_load_vector_store():
        raise FileNotFoundError("Index FAISS introuvable.")

    monkeypatch.setattr(search, "load_vector_store", fake_load_vector_store)

    with pytest.raises(FileNotFoundError, match="Index FAISS introuvable"):
        search.search_similar_events("astronomie", top_k=3)
