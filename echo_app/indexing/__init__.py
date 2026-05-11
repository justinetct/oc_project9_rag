"""Outils d'indexation pour l'application Écho."""

from echo_app.indexing.embeddings import embed_query, embed_texts, get_mistral_client


__all__ = ["embed_query", "embed_texts", "get_mistral_client"]
