"""Tests unitaires pour l'API FastAPI d'Écho.

Tous les tests sont locaux : aucun appel réseau, aucun chargement du
vector store FAISS, aucune dépendance à la clé MISTRAL_API_KEY. Le
service RAG est remplacé par un ``FakeRagService`` via
``monkeypatch.setattr`` sur l'attribut ``rag_service`` du module
``echo_app.api.main``.
"""

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from echo_app.api import main as api_main  # noqa: E402


class FakeRagService:
    """Service RAG factice pour les tests de l'API.

    Implémente uniquement ``.ask(question)`` puisque c'est tout ce que
    l'endpoint ``/ask`` utilise. Permet d'enregistrer les appels et de
    forcer une réponse ou une exception arbitraire.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.next_result: dict = {
            "question": "placeholder",
            "answer": "Réponse factice.",
            "sources": [
                {
                    "event_id": "evt-1",
                    "title": "Nuit des étoiles",
                    "city": "Paris",
                    "start_date": "2025-08-09",
                    "url": "https://example.org/evt-1",
                }
            ],
        }
        self.next_exc: Exception | None = None

    def ask(self, question: str) -> dict:
        self.calls.append(question)
        if self.next_exc is not None:
            raise self.next_exc
        return {**self.next_result, "question": question}


@pytest.fixture
def fake_service(monkeypatch):
    """Remplace le service RAG du module API par un fake."""
    fake = FakeRagService()
    monkeypatch.setattr(api_main, "rag_service", fake)
    return fake


@pytest.fixture
def client(fake_service):
    """TestClient FastAPI partageant le fake service via la fixture."""
    return TestClient(api_main.app)


def test_health_returns_ok(client):
    """GET /health retourne 200 et le payload attendu."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "echo-rag-api"}


def test_ask_returns_expected_structure(client):
    """POST /ask avec une question valide retourne 200 et la structure
    attendue (question, answer, sources)."""
    response = client.post("/ask", json={"question": "Concerts à Paris ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Concerts à Paris ?"
    assert body["answer"] == "Réponse factice."
    assert isinstance(body["sources"], list)
    assert body["sources"][0] == {
        "event_id": "evt-1",
        "title": "Nuit des étoiles",
        "city": "Paris",
        "start_date": "2025-08-09",
        "url": "https://example.org/evt-1",
    }


def test_ask_calls_rag_service(client, fake_service):
    """POST /ask délègue bien à RagService.ask() avec la bonne question."""
    client.post("/ask", json={"question": "Concerts à Paris ?"})

    assert fake_service.calls == ["Concerts à Paris ?"]


def test_ask_missing_field_returns_422(client):
    """Body sans le champ ``question`` : validation Pydantic → 422."""
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_empty_question_returns_400(client, fake_service):
    """Body avec une question vide : 400 avec un message explicite."""
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 400
    assert response.json() == {
        "detail": "La question ne peut pas être vide."
    }
    assert fake_service.calls == []


def test_ask_whitespace_question_returns_400(client, fake_service):
    """Body avec une question uniquement composée d'espaces : 400."""
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json() == {
        "detail": "La question ne peut pas être vide."
    }
    assert fake_service.calls == []


def test_ask_value_error_returns_400(client, fake_service):
    """RagService.ask qui lève ValueError : 400 + message du service."""
    fake_service.next_exc = ValueError("question invalide")

    response = client.post("/ask", json={"question": "Hello"})

    assert response.status_code == 400
    assert response.json() == {"detail": "question invalide"}


def test_ask_generic_error_returns_500(client, fake_service):
    """RagService.ask qui lève une exception inattendue : 500 générique."""
    fake_service.next_exc = RuntimeError("boom")

    response = client.post("/ask", json={"question": "Hello"})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Erreur interne pendant la génération de la réponse."
    }


def test_ask_does_not_load_vector_store(client, monkeypatch):
    """Garde-fou : aucun chargement du vector store FAISS pendant un POST /ask."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("Le vector store FAISS ne doit pas être chargé.")

    monkeypatch.setattr(
        "echo_app.indexing.langchain_faiss_store.load_langchain_vector_store",
        _boom,
    )

    response = client.post("/ask", json={"question": "Concerts ?"})

    assert response.status_code == 200


def test_ask_does_not_call_mistral(client, monkeypatch):
    """Garde-fou : aucun client Mistral n'est instancié pendant un POST /ask."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("Le client Mistral ne doit pas être appelé.")

    monkeypatch.setattr(
        "echo_app.indexing.embeddings.get_mistral_client",
        _boom,
    )

    response = client.post("/ask", json={"question": "Concerts ?"})

    assert response.status_code == 200
