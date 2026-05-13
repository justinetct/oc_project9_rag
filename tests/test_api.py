"""Tests unitaires pour l'API FastAPI d'Écho.

Tous les tests sont locaux : aucun appel réseau, aucun chargement du
vector store FAISS, aucune dépendance à la clé MISTRAL_API_KEY. Le
service RAG est remplacé par un ``FakeRagService`` via
``monkeypatch.setattr`` sur l'attribut ``rag_service`` du module
``echo_app.api.main``. La fonction ``rebuild_index`` est également
monkeypatchée dans les tests qui touchent ``POST /rebuild``.
"""

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from echo_app.api import main as api_main  # noqa: E402


class FakeRagService:
    """Service RAG factice pour les tests de l'API.

    Implémente ``.ask(question)`` et ``.reset_retriever_cache()``, et
    expose ``top_k`` pour que ``/health`` puisse le lire.
    """

    def __init__(self) -> None:
        self.top_k = 5
        self.calls: list[str] = []
        self.reset_called = False
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

    def reset_retriever_cache(self) -> None:
        self.reset_called = True


@pytest.fixture
def fake_service(monkeypatch):
    """Remplace le service RAG du module API par un fake."""
    # On remplace le vrai service RAG par un fake pour éviter les appels réels.
    fake = FakeRagService()
    monkeypatch.setattr(api_main, "rag_service", fake)
    return fake


@pytest.fixture
def client(fake_service):
    """TestClient FastAPI partageant le fake service via la fixture."""
    return TestClient(api_main.app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_enriched_payload_with_metadata(
    client, monkeypatch, tmp_path
):
    """GET /health retourne tous les champs enrichis quand l'index et la
    metadata existent."""
    (tmp_path / "index.faiss").write_text("", encoding="utf-8")
    (tmp_path / "index.pkl").write_text("", encoding="utf-8")
    # On force /health à lire un dossier temporaire contrôlé par le test.
    monkeypatch.setattr(api_main, "_resolve_vector_store_dir", lambda: tmp_path)
    # On simule une metadata de rebuild déjà présente.
    monkeypatch.setattr(
        api_main,
        "read_rebuild_metadata",
        lambda _dir: {
            "chunks_count": 155,
            "last_rebuild_at": "2026-05-13T15:42:00Z",
        },
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "echo-rag-api",
        "rag_service_ready": True,
        "vector_store_available": True,
        "chunks_count": 155,
        "top_k_default": 5,
        "last_rebuild_at": "2026-05-13T15:42:00Z",
    }


def test_health_returns_safe_defaults_when_vector_store_absent(
    client, monkeypatch, tmp_path
):
    """GET /health retourne des valeurs sûres en environnement neuf."""
    # On pointe vers un dossier vide pour simuler un environnement neuf.
    monkeypatch.setattr(api_main, "_resolve_vector_store_dir", lambda: tmp_path)
    # On simule l'absence de metadata de rebuild.
    monkeypatch.setattr(api_main, "read_rebuild_metadata", lambda _dir: None)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "echo-rag-api",
        "rag_service_ready": True,
        "vector_store_available": False,
        "chunks_count": None,
        "top_k_default": 5,
        "last_rebuild_at": None,
    }


def test_health_falls_back_to_index_mtime_when_metadata_missing(
    client, monkeypatch, tmp_path
):
    """Sans rebuild_metadata.json, last_rebuild_at provient du mtime de
    index.faiss."""
    (tmp_path / "index.faiss").write_text("", encoding="utf-8")
    (tmp_path / "index.pkl").write_text("", encoding="utf-8")
    # On utilise un vector_store temporaire avec index.faiss et index.pkl.
    monkeypatch.setattr(api_main, "_resolve_vector_store_dir", lambda: tmp_path)
    # On force le fallback sur le mtime de index.faiss.
    monkeypatch.setattr(api_main, "read_rebuild_metadata", lambda _dir: None)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["vector_store_available"] is True
    assert body["chunks_count"] is None
    assert body["last_rebuild_at"] is not None
    assert body["last_rebuild_at"].endswith("Z")


def test_health_does_not_load_vector_store(client, monkeypatch):
    """Garde-fou : /health ne charge jamais le vector store FAISS."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("Le vector store FAISS ne doit pas être chargé.")

    # Si /health charge FAISS par erreur, le test échoue.
    monkeypatch.setattr(
        "echo_app.indexing.langchain_faiss_store.load_langchain_vector_store",
        _boom,
    )

    response = client.get("/health")

    assert response.status_code == 200


def test_health_does_not_call_mistral(client, monkeypatch):
    """Garde-fou : /health n'instancie aucun client Mistral."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("Le client Mistral ne doit pas être appelé.")

    # Si /health instancie Mistral par erreur, le test échoue.
    monkeypatch.setattr(
        "echo_app.indexing.embeddings.get_mistral_client",
        _boom,
    )

    response = client.get("/health")

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------


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

    # Le fake service doit intercepter /ask avant tout chargement FAISS.
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

    # Le fake service doit éviter tout appel au client Mistral.
    monkeypatch.setattr(
        "echo_app.indexing.embeddings.get_mistral_client",
        _boom,
    )

    response = client.post("/ask", json={"question": "Concerts ?"})

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /rebuild
# ---------------------------------------------------------------------------


def _fake_rebuild_stats() -> dict:
    """Statistiques retournées par un rebuild_index mocké."""
    return {
        "documents_count": 10,
        "chunks_count": 42,
        "embeddings_count": 42,
        "embedding_dimension": 1024,
        "vectors_count": 42,
        "vector_store_path": Path("vector_store"),
        "index_file": Path("vector_store/index.faiss"),
        "docstore_file": Path("vector_store/index.pkl"),
        "last_rebuild_at": "2026-05-13T15:42:00Z",
    }


def test_rebuild_without_confirm_returns_400(client):
    """POST /rebuild sans le champ confirm : 400 avec un message clair."""
    response = client.post("/rebuild", json={})

    assert response.status_code == 400
    assert response.json() == {
        "detail": "La reconstruction de l'index nécessite confirm=true."
    }


def test_rebuild_with_confirm_false_returns_400(client):
    """POST /rebuild avec confirm=false : 400."""
    response = client.post("/rebuild", json={"confirm": False})

    assert response.status_code == 400
    assert response.json() == {
        "detail": "La reconstruction de l'index nécessite confirm=true."
    }


def test_rebuild_with_confirm_true_calls_rebuild_index(client, monkeypatch):
    """POST /rebuild avec confirm=true délègue à rebuild_index()."""
    calls: list[int] = []

    def _fake_rebuild() -> dict:
        calls.append(1)
        return _fake_rebuild_stats()

    # On remplace la vraie reconstruction par une fonction fake rapide.
    monkeypatch.setattr(api_main, "rebuild_index", _fake_rebuild)

    response = client.post("/rebuild", json={"confirm": True})

    assert response.status_code == 200
    assert calls == [1]


def test_rebuild_returns_expected_structure(client, monkeypatch):
    """POST /rebuild en succès retourne status, message, chunks_count,
    last_rebuild_at."""
    # On force une réponse de rebuild stable pour vérifier le JSON retourné.
    monkeypatch.setattr(api_main, "rebuild_index", _fake_rebuild_stats)

    response = client.post("/rebuild", json={"confirm": True})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Index reconstruit avec succès.",
        "chunks_count": 42,
        "last_rebuild_at": "2026-05-13T15:42:00Z",
    }


def test_rebuild_invalidates_retriever_cache(client, fake_service, monkeypatch):
    """Après un /rebuild réussi, RagService.reset_retriever_cache() est appelé."""
    # On simule un rebuild réussi pour tester l'invalidation du retriever.
    monkeypatch.setattr(api_main, "rebuild_index", _fake_rebuild_stats)
    assert fake_service.reset_called is False

    response = client.post("/rebuild", json={"confirm": True})

    assert response.status_code == 200
    assert fake_service.reset_called is True


def test_rebuild_failure_returns_500(client, monkeypatch):
    """Une exception levée par rebuild_index() → 500 avec message clair."""

    def _boom() -> dict:
        raise RuntimeError("embeddings down")

    # On simule une erreur de rebuild pour vérifier la réponse 500.
    monkeypatch.setattr(api_main, "rebuild_index", _boom)

    response = client.post("/rebuild", json={"confirm": True})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Erreur interne pendant la reconstruction de l'index."
    }


def test_rebuild_failure_does_not_invalidate_retriever(
    client, fake_service, monkeypatch
):
    """Si la reconstruction échoue, on n'invalide pas le retriever en cache."""

    def _boom() -> dict:
        raise RuntimeError("embeddings down")

    # On simule un échec pour vérifier que le cache n'est pas invalidé.
    monkeypatch.setattr(api_main, "rebuild_index", _boom)

    response = client.post("/rebuild", json={"confirm": True})

    assert response.status_code == 500
    assert fake_service.reset_called is False
