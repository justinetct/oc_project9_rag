"""Adaptateur LangChain pour les embeddings Mistral du projet Écho.

Cette classe expose les fonctions Mistral existantes (``embed_texts`` et
``embed_query`` dans ``echo_app.indexing.embeddings``) via l'interface
``langchain_core.embeddings.Embeddings``. Cela permet à
``langchain_community.vectorstores.FAISS`` de calculer les embeddings de
requêtes et de documents sans dupliquer la logique de batching ni la
gestion du client Mistral.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from echo_app.indexing.embeddings import embed_query, embed_texts


class MistralLangChainEmbeddings(Embeddings):
    """Adaptateur LangChain qui réutilise les embeddings Mistral du projet."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de textes (chunks) via embed_texts()."""
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        """Vectorise une requête utilisateur via embed_query()."""
        return embed_query(text)
