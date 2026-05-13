"""Tests unitaires pour la recherche sémantique sur le vector store FAISS LangChain.

Le vector store et l'adaptateur d'embeddings sont mockés via
monkeypatch : aucun appel réseau à Mistral, aucun fichier à charger.
"""

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from echo_app.indexing import search  # noqa: E402


def make_documents() -> list[Document]:
    """Construit une mini collection de Documents pour les tests."""
    return [
        Document(
            page_content="Soirée astronomie à Lanton.",
            metadata={
                "event_id": "evt-1",
                "chunk_id": "evt-1_0",
                "chunk_index": 0,
                "chunk_count": 1,
                "title": "Initiation à l'astronomie",
                "city": "Lanton",
                "url": "https://example.com/evt-1",
                "start_date": "2026-01-05T19:30",
            },
        ),
        Document(
            page_content="Concert acoustique à Andernos.",
            metadata={
                "event_id": "evt-2",
                "chunk_id": "evt-2_0",
                "chunk_index": 0,
                "chunk_count": 1,
                "title": "Concert acoustique",
                "city": "Andernos-les-Bains",
                "url": "https://example.com/evt-2",
                "start_date": "2026-06-10T20:00",
            },
        ),
        Document(
            page_content="Exposition au Bassin d'Arcachon.",
            metadata={
                "event_id": "evt-3",
                "chunk_id": "evt-3_0",
                "chunk_index": 0,
                "chunk_count": 1,
                "title": "Exposition",
                "city": "Arcachon",
                "url": "https://example.com/evt-3",
            },
        ),
    ]


class FakeVectorStore:
    """Vector store factice qui simule similarity_search_with_score()."""

    def __init__(self, documents: list[Document], distances: list[float]) -> None:
        self.documents = documents
        self.distances = distances
        self.calls: list[dict] = []

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        self.calls.append({"query": query, "k": k})
        pairs = list(zip(self.documents, self.distances, strict=True))
        return pairs[:k]


def install_fake_vector_store(
    monkeypatch,
    distances: list[float] | None = None,
) -> FakeVectorStore:
    """Installe un vector store factice et un adaptateur d'embeddings inerte."""
    documents = make_documents()
    fake_store = FakeVectorStore(
        documents=documents,
        distances=distances or [0.10, 0.25, 0.40],
    )

    def fake_load(embeddings):
        return fake_store

    class InertEmbeddings:
        def __init__(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(search, "load_langchain_vector_store", fake_load)
    monkeypatch.setattr(search, "MistralLangChainEmbeddings", InertEmbeddings)
    return fake_store


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
    # event_id et chunk_id doivent rester au niveau racine, pas dans metadata
    assert "event_id" not in first["metadata"]
    assert "chunk_id" not in first["metadata"]


def test_search_similar_events_respects_top_k(monkeypatch) -> None:
    """La taille des résultats et l'appel au vector store doivent suivre top_k."""
    fake_store = install_fake_vector_store(monkeypatch)

    results = search.search_similar_events("astronomie", top_k=1)

    assert len(results) == 1
    assert fake_store.calls == [{"query": "astronomie", "k": 1}]


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


def test_search_similar_events_result_contains_expected_keys(monkeypatch) -> None:
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
        "distance",
    }
    assert first["text"] == "Soirée astronomie à Lanton."
    assert first["score"] == pytest.approx(0.10)
    assert first["distance"] == pytest.approx(0.10)


def test_search_similar_events_propagates_missing_vector_store(monkeypatch) -> None:
    """Si le vector store est absent, l'erreur doit remonter au caller."""

    def fake_load(embeddings):
        raise FileNotFoundError("Index FAISS LangChain introuvable dans vector_store.")

    class InertEmbeddings:
        def __init__(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(search, "load_langchain_vector_store", fake_load)
    monkeypatch.setattr(search, "MistralLangChainEmbeddings", InertEmbeddings)

    with pytest.raises(FileNotFoundError, match="Index FAISS LangChain introuvable"):
        search.search_similar_events("astronomie", top_k=3)
