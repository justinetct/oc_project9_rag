"""Tests unitaires pour l'adaptateur LangChain des embeddings Mistral.

Tous les tests sont locaux : les appels à embed_texts / embed_query sont
mockés via monkeypatch, aucun appel réseau n'est effectué.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.embeddings import Embeddings  # noqa: E402

from echo_app.indexing import langchain_embeddings  # noqa: E402
from echo_app.indexing.langchain_embeddings import (  # noqa: E402
    MistralLangChainEmbeddings,
)


def test_implements_langchain_embeddings_interface() -> None:
    """L'adaptateur doit hériter de l'interface LangChain Embeddings."""
    assert isinstance(MistralLangChainEmbeddings(), Embeddings)


def test_embed_documents_delegates_to_embed_texts(monkeypatch) -> None:
    """embed_documents doit déléguer à embed_texts du projet."""
    captured: dict = {}

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        captured["texts"] = texts
        return [[0.1, 0.2], [0.3, 0.4]]

    # On remplace temporairement embed_texts par une fonction fake pour éviter
    # tout appel réel à l'API Mistral pendant le test.
    monkeypatch.setattr(langchain_embeddings, "embed_texts", fake_embed_texts)

    adapter = MistralLangChainEmbeddings()
    result = adapter.embed_documents(["Texte 1", "Texte 2"])

    assert captured["texts"] == ["Texte 1", "Texte 2"]
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_query_delegates_to_embed_query(monkeypatch) -> None:
    """embed_query doit déléguer à embed_query du projet."""
    captured: dict = {}

    def fake_embed_query(text: str) -> list[float]:
        captured["text"] = text
        return [0.5, 0.6, 0.7]

    # embed_query est remplacée par une fonction fake afin de tester 
    # uniquement la délégation de l'adaptateur LangChain.
    monkeypatch.setattr(langchain_embeddings, "embed_query", fake_embed_query)

    adapter = MistralLangChainEmbeddings()
    vector = adapter.embed_query("Concert à Arcachon ?")

    assert captured["text"] == "Concert à Arcachon ?"
    assert vector == [0.5, 0.6, 0.7]
