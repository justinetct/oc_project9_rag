"""Tests unitaires pour le service RAG d'Écho.

Tous les tests sont locaux : aucun appel réseau, aucune dépendance à la clé
MISTRAL_API_KEY. Les appels à search_similar_events et à _generate_answer
sont mockés via monkeypatch.
"""

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.rag import rag_service  # noqa: E402
from echo_app.rag.rag_service import RagService  # noqa: E402


def make_fake_results() -> list[dict]:
    """Construit trois résultats factices.

    Les deux premiers partagent le même event_id pour valider le
    dédoublonnage des sources.
    """
    return [
        {
            "text": "Soirée d'initiation à l'astronomie à Lanton.",
            "score": 0.10,
            "metadata": {
                "title": "Initiation à l'astronomie",
                "city": "Lanton",
                "url": "https://example.com/evt-1",
                "start_date": "2026-01-05T19:30",
            },
            "chunk_id": "evt-1_0",
            "event_id": "evt-1",
            "faiss_id": 0,
            "distance": 0.10,
        },
        {
            "text": "Suite de la soirée astronomie : observation du ciel.",
            "score": 0.18,
            "metadata": {
                "title": "Initiation à l'astronomie",
                "city": "Lanton",
                "url": "https://example.com/evt-1",
                "start_date": "2026-01-05T19:30",
            },
            "chunk_id": "evt-1_1",
            "event_id": "evt-1",
            "faiss_id": 1,
            "distance": 0.18,
        },
        {
            "text": "Concert acoustique à Andernos.",
            "score": 0.25,
            "metadata": {
                "title": "Concert acoustique",
                "city": "Andernos-les-Bains",
                "url": "https://example.com/evt-2",
                "start_date": "2026-06-10T20:00",
            },
            "chunk_id": "evt-2_0",
            "event_id": "evt-2",
            "faiss_id": 2,
            "distance": 0.25,
        },
    ]


def test_ask_raises_when_question_is_empty() -> None:
    """Une question vide ou composée d'espaces doit être rejetée clairement."""
    service = RagService()
    with pytest.raises(ValueError, match="question est vide"):
        service.ask("   ")


def test_build_context_numbers_chunks_and_uses_separator() -> None:
    """build_context doit numéroter les chunks et utiliser le séparateur."""
    context = rag_service.build_context(make_fake_results())

    assert "[1] Initiation à l'astronomie — Lanton" in context
    assert "[2] Initiation à l'astronomie — Lanton" in context
    assert "[3] Concert acoustique — Andernos-les-Bains" in context
    assert context.count("\n---\n") == 2
    assert "Soirée d'initiation à l'astronomie à Lanton." in context


def test_extract_sources_deduplicates_by_event_id() -> None:
    """Les sources doivent être dédoublonnées par event_id, ordre préservé."""
    sources = rag_service.extract_sources(make_fake_results())

    assert len(sources) == 2
    assert sources[0]["event_id"] == "evt-1"
    assert sources[0] == {
        "event_id": "evt-1",
        "title": "Initiation à l'astronomie",
        "city": "Lanton",
        "start_date": "2026-01-05T19:30",
        "url": "https://example.com/evt-1",
    }
    assert sources[1]["event_id"] == "evt-2"
    assert sources[1] == {
        "event_id": "evt-2",
        "title": "Concert acoustique",
        "city": "Andernos-les-Bains",
        "start_date": "2026-06-10T20:00",
        "url": "https://example.com/evt-2",
    }


def test_ask_returns_expected_structure(monkeypatch) -> None:
    """ask() doit retourner question, réponse et sources dédoublonnées."""
    captured: dict = {}

    # On remplace la recherche FAISS par une fonction locale pour éviter
    # de dépendre de l'index réel pendant le test.
    def fake_search_similar_events(question: str, top_k: int = 5) -> list[dict]:
        captured["question"] = question
        captured["top_k"] = top_k
        return make_fake_results()

    # On remplace aussi l'appel Mistral pour éviter tout appel réseau.
    # _generate_answer reçoit maintenant la liste de messages [system, user]
    # construite via LangChain.
    def fake_generate_answer(self, messages: list[dict]) -> str:
        captured["messages"] = messages
        return "Réponse simulée par le mock."

    # monkeypatch applique ces remplacements uniquement pendant ce test.
    monkeypatch.setattr(
        rag_service,
        "search_similar_events",
        fake_search_similar_events,
    )
    monkeypatch.setattr(
        RagService,
        "_generate_answer",
        fake_generate_answer,
    )

    service = RagService(top_k=3)
    response = service.ask("Quels événements d'astronomie ?")

    assert response["question"] == "Quels événements d'astronomie ?"
    assert response["answer"] == "Réponse simulée par le mock."
    assert isinstance(response["sources"], list)
    assert len(response["sources"]) == 2
    assert response["sources"][0]["title"] == "Initiation à l'astronomie"
    assert response["sources"][0]["event_id"] == "evt-1"
    assert captured["top_k"] == 3

    # Les messages produits par LangChain doivent contenir le contexte et
    # la question dans le message utilisateur.
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Contexte (extraits d'événements)" in messages[1]["content"]
    assert "Quels événements d'astronomie ?" in messages[1]["content"]


def test_ask_short_circuits_when_no_results(monkeypatch) -> None:
    """Si search retourne [], ask() ne doit pas appeler _generate_answer."""

    # Simulation d'une recherche qui ne trouve aucun chunk pertinent.
    def fake_search_similar_events(question: str, top_k: int = 5) -> list[dict]:
        return []

    # Garde-fou : si le LLM est appelé malgré l'absence de résultat, le test échoue.
    def fail_if_called(self, messages: list[dict]) -> str:
        raise AssertionError(
            "Le LLM ne doit pas être appelé lorsque la recherche est vide."
        )

    # On injecte les fonctions simulées dans le module testé.
    monkeypatch.setattr(
        rag_service,
        "search_similar_events",
        fake_search_similar_events,
    )
    monkeypatch.setattr(RagService, "_generate_answer", fail_if_called)

    service = RagService()
    response = service.ask("événement complètement inconnu xyz")

    assert response["question"] == "événement complètement inconnu xyz"
    assert response["sources"] == []
    assert "aucun événement" in response["answer"].lower()


def test_constructor_rejects_invalid_top_k() -> None:
    """top_k doit rester strictement positif."""
    with pytest.raises(ValueError, match="strictement positif"):
        RagService(top_k=0)
