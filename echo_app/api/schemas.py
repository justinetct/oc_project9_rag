"""Schémas Pydantic de l'API FastAPI d'Écho.

Ces schémas définissent la forme des requêtes et réponses JSON exposées
par l'API. Ils sont volontairement simples : pas de contrainte de
longueur sur ``question`` (la validation vide/whitespace est faite dans
l'endpoint pour renvoyer un 400 explicite), et tous les champs d'une
source sont optionnels car ils proviennent de ``metadata.get(...)``
côté ``RagService``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Corps de la requête POST /ask."""

    question: str = Field(
        ...,
        description="Question utilisateur à poser au chatbot Écho.",
        examples=["Quels événements autour de l astronomie sont proposés ?"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quels événements autour de l astronomie sont proposés ?"
            }
        }
    }


SOURCE_EXAMPLE = {
    "event_id": "evt-nuit-des-etoiles-2025",
    "title": "Nuit des étoiles à l'Observatoire de Paris",
    "city": "Paris",
    "start_date": "2025-08-09",
    "url": "https://openagenda.com/echo/events/nuit-des-etoiles-2025",
}


class SourceResponse(BaseModel):
    """Une source affichable retournée par le service RAG."""

    event_id: str | None = None
    title: str | None = None
    city: str | None = None
    start_date: str | None = None
    url: str | None = None

    model_config = {"json_schema_extra": {"example": SOURCE_EXAMPLE}}


class AskResponse(BaseModel):
    """Réponse de l'endpoint POST /ask."""

    question: str
    answer: str
    sources: list[SourceResponse]

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quels événements autour de l astronomie sont proposés ?",
                "answer": (
                    "Plusieurs événements autour de l'astronomie sont "
                    "programmés cet été : la Nuit des étoiles à l'Observatoire "
                    "de Paris le 9 août 2025, ainsi qu'une conférence "
                    "« Découvrir le ciel d'été » à Bordeaux le 12 août 2025."
                ),
                "sources": [
                    SOURCE_EXAMPLE,
                    {
                        "event_id": "evt-conf-ciel-ete-2025",
                        "title": "Conférence : Découvrir le ciel d'été",
                        "city": "Bordeaux",
                        "start_date": "2025-08-12",
                        "url": "https://openagenda.com/echo/events/ciel-d-ete",
                    },
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    """Réponse de l'endpoint GET /health.

    Voulu minimal et rapide : confirme uniquement que l'API répond et
    que le service RAG est instancié. Les détails techniques (vector
    store, modèles, etc.) sont exposés par ``GET /metadata``.
    """

    status: str
    service: str
    rag_service_ready: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "service": "echo-rag-api",
                "rag_service_ready": True,
            }
        }
    }


class MetadataResponse(BaseModel):
    """Réponse de l'endpoint GET /metadata.

    Donne les informations techniques non sensibles utiles pour
    l'exploitation et la démo : état du service RAG, disponibilité du
    vector store, statistiques d'indexation et noms des modèles Mistral
    utilisés. Ne contient aucun secret ni chemin local absolu.
    """

    service: str
    rag_service_ready: bool
    vector_store_available: bool
    chunks_count: int | None = None
    top_k_default: int
    last_rebuild_at: str | None = None
    embedding_model: str
    generation_model: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "service": "echo-rag-api",
                "rag_service_ready": True,
                "vector_store_available": True,
                "chunks_count": 155,
                "top_k_default": 5,
                "last_rebuild_at": "2026-05-13T15:42:00Z",
                "embedding_model": "mistral-embed",
                "generation_model": "mistral-small-latest",
            }
        }
    }


class RebuildRequest(BaseModel):
    """Corps de la requête POST /rebuild.

    La reconstruction de l'index est une opération coûteuse : elle doit
    être confirmée explicitement avec ``confirm=true``.
    """

    confirm: bool = Field(
        default=False,
        description="Doit être ``true`` pour déclencher la reconstruction.",
        examples=[True],
    )

    model_config = {"json_schema_extra": {"example": {"confirm": True}}}


class RebuildResponse(BaseModel):
    """Réponse de l'endpoint POST /rebuild en cas de succès."""

    status: str
    message: str
    chunks_count: int
    last_rebuild_at: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "message": "Index reconstruit avec succès.",
                "chunks_count": 155,
                "last_rebuild_at": "2026-05-13T15:42:00Z",
            }
        }
    }
