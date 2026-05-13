"""API FastAPI d'Écho.

Cette API expose le service RAG existant via trois endpoints HTTP :

- ``GET /health`` : état de l'API, du service RAG et du vector store.
- ``POST /ask`` : pose une question au système RAG et retourne la
  réponse générée ainsi que les sources utilisées.
- ``POST /rebuild`` : reconstruit localement l'index vectoriel
  (réservé au POC, demande une confirmation explicite).

L'API ne réimplémente aucune logique RAG : elle se contente d'appeler
``RagService.ask(question)`` et de retourner sa réponse en JSON. La
reconstruction de l'index est portée par
``echo_app.indexing.rebuild.rebuild_index`` afin d'être testable et
réutilisée par le script CLI ``scripts/rebuild_index.py``.

Lancement local :

    poetry run uvicorn echo_app.api.main:app --reload

Documentation Swagger : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from echo_app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RebuildRequest,
    RebuildResponse,
)
from echo_app.indexing.langchain_faiss_store import (
    FAISS_DOCSTORE_FILENAME,
    FAISS_INDEX_FILENAME,
    VECTOR_STORE_DIR,
)
from echo_app.indexing.rebuild import (
    read_rebuild_metadata,
    rebuild_index,
)
from echo_app.rag.rag_service import RagService
from src.config import PATHS

app = FastAPI(
    title="Écho - API RAG",
    description="API FastAPI pour interroger le chatbot culturel Écho.",
    version="0.1.0",
)

rag_service = RagService()


def _resolve_vector_store_dir() -> Path:
    """Résout le dossier du vector store depuis la racine du projet."""
    if VECTOR_STORE_DIR.is_absolute():
        return VECTOR_STORE_DIR
    return PATHS.root / VECTOR_STORE_DIR


def _collect_health_info() -> dict:
    """Construit les informations exposées par ``GET /health``.

    Ne charge pas le retriever, ne contacte ni FAISS ni Mistral. Si le
    vector store n'existe pas (environnement neuf, CI), retourne des
    valeurs sûres : ``vector_store_available=False``, ``chunks_count``
    et ``last_rebuild_at`` à ``None``.
    """
    vector_store_dir = _resolve_vector_store_dir()
    index_file = vector_store_dir / FAISS_INDEX_FILENAME
    docstore_file = vector_store_dir / FAISS_DOCSTORE_FILENAME
    vector_store_available = index_file.exists() and docstore_file.exists()

    metadata = read_rebuild_metadata(VECTOR_STORE_DIR)
    chunks_count: int | None = None
    last_rebuild_at: str | None = None
    if metadata is not None:
        chunks_count = metadata.get("chunks_count")
        last_rebuild_at = metadata.get("last_rebuild_at")

    if last_rebuild_at is None and index_file.exists():
        index_mtime = datetime.fromtimestamp(
            index_file.stat().st_mtime, tz=timezone.utc
        )
        last_rebuild_at = index_mtime.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "status": "ok",
        "service": "echo-rag-api",
        "rag_service_ready": rag_service is not None,
        "vector_store_available": vector_store_available,
        "chunks_count": chunks_count,
        "top_k_default": rag_service.top_k,
        "last_rebuild_at": last_rebuild_at,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Endpoint de santé : état de l'API et du vector store."""
    return HealthResponse(**_collect_health_info())


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


@app.post("/rebuild", response_model=RebuildResponse)
def rebuild(request: RebuildRequest) -> RebuildResponse:
    """Reconstruit localement le vector store FAISS LangChain.

    L'opération est synchrone et peut être longue : elle est réservée au
    POC local. Elle exige ``confirm=true`` dans le corps de la requête.
    Après reconstruction, le retriever en cache du service RAG est
    invalidé afin que le prochain ``/ask`` reparte du nouvel index.
    """
    if request.confirm is not True:
        raise HTTPException(
            status_code=400,
            detail="La reconstruction de l'index nécessite confirm=true.",
        )

    try:
        stats = rebuild_index()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la reconstruction de l'index.",
        ) from exc

    rag_service.reset_retriever_cache()

    return RebuildResponse(
        status="ok",
        message="Index reconstruit avec succès.",
        chunks_count=stats["chunks_count"],
        last_rebuild_at=stats.get("last_rebuild_at"),
    )
