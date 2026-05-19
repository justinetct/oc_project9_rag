"""Tests unitaires pour le service RAG d'Écho.

Tous les tests sont locaux : aucun appel réseau, aucune dépendance à la
clé MISTRAL_API_KEY. Le retriever LangChain et l'appel Mistral sont
mockés via monkeypatch (le retriever est injecté directement sur
l'instance ``RagService``).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from echo_app.rag import rag_service  # noqa: E402
from echo_app.rag.rag_service import RagService  # noqa: E402


def _make_chat_response(content: str) -> SimpleNamespace:
    """Construit une réponse Mistral factice avec ``choices[0].message.content``."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_capacity_error() -> Exception:
    """Construit une exception identique à celle observée en production."""
    return RuntimeError(
        "API error occurred: Status 429. Body: "
        '{"object":"error","message":"Service tier capacity exceeded '
        'for this model.","type":"service_tier_capacity_exceeded",'
        '"code":"3505","raw_status_code":429}'
    )


class FakeRetriever:
    """Retriever minimal compatible LangChain pour les tests.

    Implémente uniquement ``.invoke(query)`` car c'est tout ce que
    ``RagService.ask()`` utilise.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.calls.append(query)
        return self.documents


def make_fake_documents() -> list[Document]:
    """Construit trois Documents factices.

    Les deux premiers partagent le même event_id pour valider le
    dédoublonnage des sources.
    """
    return [
        Document(
            page_content="Soirée d'initiation à l'astronomie à Lanton.",
            metadata={
                "event_id": "evt-1",
                "chunk_id": "evt-1_0",
                "title": "Initiation à l'astronomie",
                "city": "Lanton",
                "url": "https://example.com/evt-1",
                "start_date": "2026-01-05T19:30",
            },
        ),
        Document(
            page_content="Suite de la soirée astronomie : observation du ciel.",
            metadata={
                "event_id": "evt-1",
                "chunk_id": "evt-1_1",
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
                "title": "Concert acoustique",
                "city": "Andernos-les-Bains",
                "url": "https://example.com/evt-2",
                "start_date": "2026-06-10T20:00",
            },
        ),
    ]


def test_ask_raises_when_question_is_empty() -> None:
    """Une question vide ou composée d'espaces doit être rejetée clairement."""
    service = RagService()
    service._retriever = FakeRetriever(documents=make_fake_documents())
    with pytest.raises(ValueError, match="question est vide"):
        service.ask("   ")


def test_build_context_numbers_chunks_and_uses_separator() -> None:
    """build_context doit numéroter les chunks et utiliser le séparateur."""
    context = rag_service.build_context(make_fake_documents())

    assert "[1] Initiation à l'astronomie — Lanton" in context
    assert "[2] Initiation à l'astronomie — Lanton" in context
    assert "[3] Concert acoustique — Andernos-les-Bains" in context
    assert context.count("\n---\n") == 2
    assert "Soirée d'initiation à l'astronomie à Lanton." in context


def test_extract_sources_deduplicates_by_event_id() -> None:
    """Les sources doivent être dédoublonnées par event_id, ordre préservé."""
    sources = rag_service.extract_sources(make_fake_documents())

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

    # On remplace l'appel Mistral pour éviter tout appel réseau.
    # _generate_answer reçoit la liste de messages [system, user]
    # construite via LangChain.
    def fake_generate_answer(self, messages: list[dict]) -> str:
        captured["messages"] = messages
        return "Réponse simulée par le mock."

    monkeypatch.setattr(RagService, "_generate_answer", fake_generate_answer)

    fake_retriever = FakeRetriever(documents=make_fake_documents())
    service = RagService(top_k=3)
    service._retriever = fake_retriever

    response = service.ask("Quels événements d'astronomie ?")

    assert response["question"] == "Quels événements d'astronomie ?"
    assert response["answer"] == "Réponse simulée par le mock."
    assert isinstance(response["sources"], list)
    assert len(response["sources"]) == 2
    assert response["sources"][0]["title"] == "Initiation à l'astronomie"
    assert response["sources"][0]["event_id"] == "evt-1"
    assert fake_retriever.calls == ["Quels événements d'astronomie ?"]

    # Les messages produits par LangChain doivent contenir le contexte
    # et la question dans le message utilisateur.
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Contexte (extraits d'événements)" in messages[1]["content"]
    assert "Quels événements d'astronomie ?" in messages[1]["content"]


def test_ask_short_circuits_when_no_documents(monkeypatch) -> None:
    """Si le retriever retourne [], ask() ne doit pas appeler _generate_answer."""

    # Garde-fou : si le LLM est appelé malgré l'absence de résultat, le test échoue.
    def fail_if_called(self, messages: list[dict]) -> str:
        raise AssertionError(
            "Le LLM ne doit pas être appelé lorsque le retriever renvoie [] ."
        )

    monkeypatch.setattr(RagService, "_generate_answer", fail_if_called)

    service = RagService()
    service._retriever = FakeRetriever(documents=[])

    response = service.ask("événement complètement inconnu xyz")

    assert response["question"] == "événement complètement inconnu xyz"
    assert response["sources"] == []
    assert "aucun événement" in response["answer"].lower()


def test_constructor_rejects_invalid_top_k() -> None:
    """top_k doit rester strictement positif."""
    with pytest.raises(ValueError, match="strictement positif"):
        RagService(top_k=0)


def test_ask_with_include_contexts_returns_raw_chunks(monkeypatch) -> None:
    """include_contexts=True ajoute les page_content bruts pour Ragas."""

    def fake_generate_answer(self, messages: list[dict]) -> str:
        return "Réponse simulée par le mock."

    monkeypatch.setattr(RagService, "_generate_answer", fake_generate_answer)

    service = RagService(top_k=3)
    service._retriever = FakeRetriever(documents=make_fake_documents())

    response = service.ask(
        "Quels événements d'astronomie ?", include_contexts=True
    )

    assert response["question"] == "Quels événements d'astronomie ?"
    assert response["answer"] == "Réponse simulée par le mock."
    assert "contexts" in response
    assert response["contexts"] == [
        "Soirée d'initiation à l'astronomie à Lanton.",
        "Suite de la soirée astronomie : observation du ciel.",
        "Concert acoustique à Andernos.",
    ]


def test_ask_with_include_contexts_empty_retrieval_returns_empty_list() -> None:
    """include_contexts=True avec retriever vide doit produire contexts=[]."""
    service = RagService()
    service._retriever = FakeRetriever(documents=[])

    response = service.ask("question hors sujet xyz", include_contexts=True)

    assert response["sources"] == []
    assert response["contexts"] == []


def test_ask_default_does_not_expose_contexts(monkeypatch) -> None:
    """Comportement par défaut inchangé : pas de clé `contexts` dans le dict."""

    def fake_generate_answer(self, messages: list[dict]) -> str:
        return "Réponse simulée par le mock."

    monkeypatch.setattr(RagService, "_generate_answer", fake_generate_answer)

    service = RagService(top_k=3)
    service._retriever = FakeRetriever(documents=make_fake_documents())

    response = service.ask("Quels événements d'astronomie ?")

    assert set(response.keys()) == {"question", "answer", "sources"}


def test_generate_answer_retries_on_mistral_capacity_error(monkeypatch) -> None:
    """Un 429 capacity au 1er appel doit déclencher un retry qui réussit."""
    calls = {"count": 0}

    def fake_complete(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _make_capacity_error()
        return _make_chat_response("Réponse après retry.")

    fake_client = SimpleNamespace(chat=SimpleNamespace(complete=fake_complete))
    monkeypatch.setattr(rag_service, "get_mistral_client", lambda: fake_client)
    # On neutralise le sleep pour ne pas ralentir les tests.
    monkeypatch.setattr(rag_service.time, "sleep", lambda _: None)

    service = RagService()
    answer = service._generate_answer([{"role": "user", "content": "hi"}])

    assert answer == "Réponse après retry."
    assert calls["count"] == 2


def test_generate_answer_does_not_retry_on_other_error(monkeypatch) -> None:
    """Une erreur qui n'est pas un 429 capacity doit remonter sans retry."""
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_complete(**kwargs):
        calls["count"] += 1
        raise ValueError("Erreur de parsing JSON")

    fake_client = SimpleNamespace(chat=SimpleNamespace(complete=fake_complete))
    monkeypatch.setattr(rag_service, "get_mistral_client", lambda: fake_client)
    monkeypatch.setattr(
        rag_service.time, "sleep", lambda delay: sleeps.append(delay)
    )

    service = RagService()
    with pytest.raises(ValueError, match="parsing JSON"):
        service._generate_answer([{"role": "user", "content": "hi"}])

    assert calls["count"] == 1
    assert sleeps == []


def test_generate_answer_raises_after_two_capacity_errors(monkeypatch) -> None:
    """Deux 429 capacity successives → l'exception remonte après le retry."""
    calls = {"count": 0}

    def fake_complete(**kwargs):
        calls["count"] += 1
        raise _make_capacity_error()

    fake_client = SimpleNamespace(chat=SimpleNamespace(complete=fake_complete))
    monkeypatch.setattr(rag_service, "get_mistral_client", lambda: fake_client)
    monkeypatch.setattr(rag_service.time, "sleep", lambda _: None)

    service = RagService()
    with pytest.raises(RuntimeError, match="service_tier_capacity_exceeded"):
        service._generate_answer([{"role": "user", "content": "hi"}])

    assert calls["count"] == 2
