"""Génération des embeddings Mistral pour les chunks et les requêtes."""

from __future__ import annotations

import os

from echo_app.config import get_mistral_api_key

try:
    from mistralai import Mistral
except ImportError:  # pragma: no cover - compatibilité avec certaines versions du SDK
    from mistralai.client import Mistral


DEFAULT_EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIMENSION = 1024


def _get_embedding_model() -> str:
    """Lit le modèle d'embedding depuis l'environnement, avec une valeur par défaut."""
    return os.getenv("MISTRAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _filter_valid_texts(texts: list[str]) -> list[str]:
    """Garde uniquement les textes non vides avant l'appel à l'API.

    Les chunks sont déjà contrôlés au moment du chunking. Cette fonction sert
    seulement de sécurité locale pour éviter d'envoyer une chaîne vide à Mistral.
    """
    valid_texts: list[str] = []

    for text in texts:
        if text is None:
            continue

        text = str(text).strip()
        if text:
            valid_texts.append(text)

    if not valid_texts:
        raise ValueError("Aucun texte valide à vectoriser.")

    return valid_texts


def _batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Regroupe les textes par lots pour limiter le nombre d'appels API.

    Un chunk est un morceau de document à vectoriser.
    Un batch est seulement un paquet technique de plusieurs chunks envoyé à Mistral.
    """
    if batch_size <= 0:
        raise ValueError("batch_size doit être positif.")

    batches: list[list[str]] = []
    for start_index in range(0, len(texts), batch_size):
        batches.append(texts[start_index : start_index + batch_size])

    return batches


def _extract_embeddings(response, expected_count: int) -> list[list[float]]:
    """Récupère les vecteurs Mistral et vérifie leur cohérence.

    Chaque texte envoyé doit produire exactement un vecteur de dimension 1024.
    Cette vérification évite de construire plus tard un index FAISS incohérent.
    """
    data = getattr(response, "data", None)
    if not data:
        raise RuntimeError("La réponse Mistral ne contient aucun embedding.")

    # Certaines réponses contiennent un index : on l'utilise pour préserver l'ordre
    # entre les textes envoyés et les embeddings retournés.
    if all(getattr(item, "index", None) is not None for item in data):
        data = sorted(data, key=lambda item: item.index)

    embeddings: list[list[float]] = []

    for item in data:
        embedding = getattr(item, "embedding", None)
        if not embedding:
            raise RuntimeError("La réponse Mistral contient un embedding vide.")

        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Dimension d'embedding incohérente : "
                f"{len(embedding)} au lieu de {EMBEDDING_DIMENSION}."
            )

        embeddings.append(list(embedding))

    if len(embeddings) != expected_count:
        raise RuntimeError("Le nombre d'embeddings retournés par Mistral est incohérent.")

    return embeddings


def get_mistral_client(api_key: str | None = None) -> Mistral:
    """Crée le client Mistral sans jamais stocker la clé API dans le code."""
    resolved_api_key = api_key or get_mistral_api_key()

    if not str(resolved_api_key).strip():
        raise ValueError("La clé API Mistral est manquante.")

    return Mistral(api_key=resolved_api_key)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Transforme une liste de textes en vecteurs numériques.

    Cette fonction est utilisée pour vectoriser les chunks avant l'indexation FAISS.
    Les textes sont envoyés à Mistral par batchs pour éviter un appel API par chunk.
    """
    valid_texts = _filter_valid_texts(texts)
    client = get_mistral_client()
    model_name = _get_embedding_model()

    all_embeddings: list[list[float]] = []

    for batch in _batch_texts(valid_texts, batch_size):
        try:
            response = client.embeddings.create(model=model_name, inputs=batch)
        except Exception as exc:  # pragma: no cover - dépend du SDK et du réseau
            raise RuntimeError("La requête d'embeddings Mistral a échoué.") from exc

        batch_embeddings = _extract_embeddings(response, expected_count=len(batch))
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Transforme une question utilisateur en un seul vecteur de recherche."""
    return embed_texts([query])[0]
