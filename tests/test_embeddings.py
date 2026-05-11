"""Tests unitaires pour les embeddings Mistral."""

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.indexing import embeddings


class FakeEmbeddingItem:
    """Objet simple qui simule un embedding retourné par le SDK."""

    def __init__(self, embedding: list[float], index: int | None = None) -> None:
        self.embedding = embedding
        self.index = index


class FakeEmbeddingResponse:
    """Réponse simple qui simule la structure du SDK Mistral."""

    def __init__(self, data: list[FakeEmbeddingItem]) -> None:
        self.data = data


class FakeEmbeddingsAPI:
    """Sous-client factice qui enregistre les appels à create()."""

    def __init__(self, embedding_dimension: int) -> None:
        self.embedding_dimension = embedding_dimension
        self.calls: list[dict] = []

    def create(self, *, model: str, inputs: list[str]) -> FakeEmbeddingResponse:
        self.calls.append({"model": model, "inputs": inputs})
        data = [
            FakeEmbeddingItem(
                embedding=[float(index)] * self.embedding_dimension,
                index=index,
            )
            for index, _text in enumerate(inputs)
        ]
        return FakeEmbeddingResponse(data)


class FakeMistralClient:
    """Client factice qui expose uniquement l'API embeddings utile ici."""

    def __init__(self, embedding_dimension: int = 1024) -> None:
        self.embeddings = FakeEmbeddingsAPI(embedding_dimension=embedding_dimension)


def test_filter_valid_texts_ignores_empty_values() -> None:
    """Les textes vides doivent être ignorés sans casser l'ordre."""
    texts = [" Bonjour ", "", "   ", "\nTexte 2\n", None]

    cleaned_texts = embeddings._filter_valid_texts(texts)

    assert cleaned_texts == ["Bonjour", "Texte 2"]


def test_batch_texts_splits_input_into_simple_batches() -> None:
    """Le découpage en batchs doit rester simple et prévisible."""
    texts = ["a", "b", "c", "d", "e"]

    batches = embeddings._batch_texts(texts, batch_size=2)

    assert batches == [["a", "b"], ["c", "d"], ["e"]]


def test_embed_query_calls_embed_texts_with_single_query(monkeypatch) -> None:
    """embed_query doit déléguer à embed_texts avec une seule requête."""
    captured: dict = {}

    def fake_embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
        captured["texts"] = texts
        captured["batch_size"] = batch_size
        return [[0.1] * embeddings.EMBEDDING_DIMENSION]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)

    vector = embeddings.embed_query("Que voir à Arcachon ?")

    assert captured["texts"] == ["Que voir à Arcachon ?"]
    assert captured["batch_size"] == 32
    assert len(vector) == embeddings.EMBEDDING_DIMENSION


def test_embed_texts_raises_when_no_valid_text_is_provided() -> None:
    """Une erreur claire doit être levée si aucun texte n'est exploitable."""
    with pytest.raises(ValueError, match="Aucun texte valide"):
        embeddings.embed_texts(["", "   ", "\n"])


def test_extract_embeddings_raises_when_dimension_is_not_1024() -> None:
    """Une dimension différente de 1024 doit être rejetée."""
    response = FakeEmbeddingResponse(
        [FakeEmbeddingItem(embedding=[0.1] * 12, index=0)]
    )

    with pytest.raises(ValueError, match="Dimension d'embedding incohérente"):
        embeddings._extract_embeddings(response, expected_count=1)


def test_embed_texts_keeps_valid_order_and_uses_batches(monkeypatch) -> None:
    """Les textes valides doivent garder leur ordre et être envoyés par batch."""
    fake_client = FakeMistralClient()

    monkeypatch.setattr(embeddings, "get_mistral_client", lambda api_key=None: fake_client)
    monkeypatch.setenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")

    vectors = embeddings.embed_texts(
        [" Premier ", "", "Deuxième", "   ", "Troisième "],
        batch_size=2,
    )

    assert len(vectors) == 3
    assert all(len(vector) == embeddings.EMBEDDING_DIMENSION for vector in vectors)
    assert fake_client.embeddings.calls == [
        {"model": "mistral-embed", "inputs": ["Premier", "Deuxième"]},
        {"model": "mistral-embed", "inputs": ["Troisième"]},
    ]


def test_get_mistral_client_raises_when_api_key_is_missing(monkeypatch) -> None:
    """Une erreur claire doit être levée si la clé API est absente."""
    monkeypatch.setattr(embeddings, "get_mistral_api_key", lambda: "")

    with pytest.raises(ValueError, match="clé API Mistral est manquante"):
        embeddings.get_mistral_client()
