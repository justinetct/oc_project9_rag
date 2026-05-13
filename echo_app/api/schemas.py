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


class SourceResponse(BaseModel):
    """Une source affichable retournée par le service RAG."""

    event_id: str | None = None
    title: str | None = None
    city: str | None = None
    start_date: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    """Réponse de l'endpoint POST /ask."""

    question: str
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    """Réponse de l'endpoint GET /health.

    Donne quelques informations utiles sur l'état du service RAG et du
    vector store local, sans déclencher de chargement coûteux.
    """

    status: str
    service: str
    rag_service_ready: bool
    vector_store_available: bool
    chunks_count: int | None = None
    top_k_default: int
    last_rebuild_at: str | None = None


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


class RebuildResponse(BaseModel):
    """Réponse de l'endpoint POST /rebuild en cas de succès."""

    status: str
    message: str
    chunks_count: int
    last_rebuild_at: str | None = None
