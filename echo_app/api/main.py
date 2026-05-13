"""API FastAPI d'Écho.

Cette API expose le service RAG existant via deux endpoints HTTP :

- ``GET /health`` : vérifie que l'API répond.
- ``POST /ask`` : pose une question au système RAG et retourne la
  réponse générée ainsi que les sources utilisées.

L'API ne réimplémente aucune logique RAG : elle se contente d'appeler
``RagService.ask(question)`` et de retourner sa réponse en JSON. Le
service est instancié au chargement du module ; le chargement effectif
de l'index FAISS et du client Mistral reste différé jusqu'au premier
``ask`` (cf. ``RagService``).

Lancement local :

    poetry run uvicorn echo_app.api.main:app --reload

Documentation Swagger : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from echo_app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
)
from echo_app.rag.rag_service import RagService

app = FastAPI(
    title="Écho - API RAG",
    description="API FastAPI pour interroger le chatbot culturel Écho.",
    version="0.1.0",
)

rag_service = RagService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Endpoint de santé : confirme que l'API répond."""
    return HealthResponse(status="ok", service="echo-rag-api")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Pose une question au système RAG et retourne la réponse + sources."""
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide.",
        )
    try:
        result = rag_service.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la génération de la réponse.",
        ) from exc
    return AskResponse(**result)
