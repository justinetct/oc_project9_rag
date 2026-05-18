"""API FastAPI d'Écho.

Cette API expose le service RAG existant via quatre endpoints HTTP :

- ``GET /health`` : état minimal de l'API (rapide, sans I/O).
- ``GET /metadata`` : informations techniques sur le service RAG, le
  vector store et les modèles Mistral (sans secret).
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

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from echo_app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
    RebuildRequest,
    RebuildResponse,
)
from echo_app.config import MISTRAL_MODEL
from echo_app.indexing.embeddings import DEFAULT_EMBEDDING_MODEL
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


EMBEDDING_MODEL_NAME = os.getenv("MISTRAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
GENERATION_MODEL_NAME = MISTRAL_MODEL


# On utilise le logger Uvicorn déjà configuré pour que ces messages
# apparaissent dans la même stream que les logs serveur (utile en Docker).
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Affiche les URL utiles pour la démo locale au démarrage."""
    logger.info("Écho API prête : http://127.0.0.1:8000/docs")
    logger.info("Health check : http://127.0.0.1:8000/health")
    yield


API_DESCRIPTION = """
**Écho** est un chatbot culturel développé par **Puls-Events** pour aider les
utilisateurs à découvrir des événements culturels autour du **Bassin d'Arcachon**.

Le corpus est construit à partir des événements publics **OpenAgenda** (via
Opendatasoft) : collecte, filtrage géographique et temporel, nettoyage et
transformation en documents Markdown, découpage en chunks puis indexation dans
un vector store **FAISS** via **LangChain**. Les embeddings et la génération
de réponse s'appuient sur **Mistral AI** (`mistral-embed` pour l'indexation,
`mistral-small-latest` pour la génération).

Cette API expose le service RAG existant sans réimplémenter la logique :
elle se contente d'appeler `RagService.ask(question)` et de renvoyer
`{question, answer, sources}`. L'index FAISS est chargé en mémoire au premier
`POST /ask` et conservé en cache pour les requêtes suivantes.

### Endpoints

- **`GET /health`** — état minimal de l'API (rapide, sans I/O).
- **`GET /metadata`** — informations techniques sur le service RAG, le vector
  store et les modèles Mistral utilisés (aucun secret exposé).
- **`POST /ask`** — pose une question et retourne la réponse générée par
  Mistral, accompagnée des événements sources retrouvés dans l'index.
- **`POST /rebuild`** — reconstruit localement l'index FAISS à partir des
  documents préparés ; réservé au POC local, exige `confirm=true`.

Cet endpoint `/rebuild` n'est pas pensé pour la production : il devrait y être
protégé (authentification, rôle dédié) ou remplacé par une tâche planifiée.
""".strip()


app = FastAPI(
    title="Écho - API RAG",
    description=API_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)

rag_service = RagService()


def _error_example(detail) -> dict:
    """Construit une entrée ``responses`` Swagger avec un exemple concret."""
    return {
        "content": {"application/json": {"example": {"detail": detail}}}
    }


ASK_RESPONSES: dict = {
    400: {
        "description": "Question vide ou composée uniquement d'espaces.",
        **_error_example("La question ne peut pas être vide."),
    },
    422: {
        "description": "Le corps JSON est invalide ou le champ ``question`` est manquant.",
        **_error_example(
            [
                {
                    "type": "missing",
                    "loc": ["body", "question"],
                    "msg": "Field required",
                    "input": {},
                }
            ],
        ),
    },
    500: {
        "description": "Erreur interne pendant la génération de la réponse.",
        **_error_example("Erreur interne pendant la génération de la réponse."),
    },
}

REBUILD_RESPONSES: dict = {
    400: {
        "description": "La reconstruction n'a pas été confirmée (``confirm`` absent ou ``false``).",
        **_error_example("La reconstruction de l'index nécessite confirm=true."),
    },
    500: {
        "description": "Erreur interne pendant la reconstruction de l'index.",
        **_error_example("Erreur interne pendant la reconstruction de l'index."),
    },
}


def _resolve_vector_store_dir() -> Path:
    """Résout le dossier du vector store depuis la racine du projet."""
    if VECTOR_STORE_DIR.is_absolute():
        return VECTOR_STORE_DIR
    return PATHS.root / VECTOR_STORE_DIR


def _collect_metadata_info() -> dict:
    """Construit les informations exposées par ``GET /metadata``.

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
        "service": "echo-rag-api",
        "rag_service_ready": rag_service is not None,
        "vector_store_available": vector_store_available,
        "chunks_count": chunks_count,
        "top_k_default": rag_service.top_k,
        "last_rebuild_at": last_rebuild_at,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "generation_model": GENERATION_MODEL_NAME,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """État minimal de l'API.

    **Paramètres :** aucun.

    **Réponse (200) :**
    - `status` : toujours `"ok"` si l'API répond.
    - `service` : identifiant du service (`"echo-rag-api"`).
    - `rag_service_ready` : `true` si le service RAG est instancié côté API.

    Aucune I/O : pas de chargement FAISS, pas d'appel Mistral. Les détails
    techniques (vector store, modèles, etc.) sont exposés par
    `GET /metadata`.
    """
    return HealthResponse(
        status="ok",
        service="echo-rag-api",
        rag_service_ready=rag_service is not None,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Informations techniques sur le service RAG et le vector store.

    **Paramètres :** aucun.

    **Réponse (200) :**
    - `service` : identifiant du service (`"echo-rag-api"`).
    - `rag_service_ready` : `true` si le service RAG est instancié côté API.
    - `vector_store_available` : `true` si `vector_store/index.faiss` et
      `vector_store/index.pkl` sont présents sur le disque.
    - `chunks_count` : nombre de chunks indexés (lu dans
      `vector_store/rebuild_metadata.json`, ou `null` si la metadata est
      absente).
    - `top_k_default` : valeur de `top_k` utilisée par défaut par `RagService`.
    - `last_rebuild_at` : horodatage UTC ISO 8601 de la dernière reconstruction
      via l'API, ou à défaut le `mtime` de `index.faiss`. `null` si aucun index
      n'existe sur le disque.
    - `embedding_model` : nom du modèle Mistral d'embeddings utilisé.
    - `generation_model` : nom du modèle Mistral de génération utilisé.

    Aucune I/O coûteuse : lit uniquement les métadonnées du vector store sur
    disque et expose les noms des modèles configurés. **Aucun secret (clé API)
    ni chemin local absolu n'est exposé.**
    """
    return MetadataResponse(**_collect_metadata_info())


@app.post("/ask", response_model=AskResponse, responses=ASK_RESPONSES)
def ask(request: AskRequest) -> AskResponse:
    """Pose une question au système RAG et retourne la réponse générée.

    **Paramètres (corps JSON) :**
    - `question` *(str, obligatoire)* : la question utilisateur en français.

    **Réponse (200) :**
    - `question` : la question reçue (renvoyée telle quelle).
    - `answer` : la réponse générée par Mistral à partir des chunks pertinents.
      Si aucun chunk n'est retrouvé, la réponse indique l'absence de résultat.
    - `sources` : liste dédoublonnée par `event_id` des événements ayant servi
      à la génération (`event_id`, `title`, `city`, `start_date`, `url`).

    **Codes d'erreur :**
    - `400` : `question` vide ou composée uniquement d'espaces.
    - `422` : champ `question` manquant ou JSON invalide.
    - `500` : erreur inattendue côté service RAG (ex. embeddings Mistral
      indisponibles).

    L'index FAISS est chargé en mémoire au premier appel et conservé en cache
    pour les requêtes suivantes : aucune reconstruction par requête.
    """
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


@app.post("/rebuild", response_model=RebuildResponse, responses=REBUILD_RESPONSES)
def rebuild(request: RebuildRequest) -> RebuildResponse:
    """Reconstruit localement le vector store FAISS LangChain.

    Opération coûteuse (lit les documents, génère des embeddings Mistral pour
    chaque chunk, reconstruit l'index). Réservée au POC local.

    **Paramètres (corps JSON) :**
    - `confirm` *(bool, obligatoire)* : doit valoir `true` pour déclencher la
      reconstruction. Toute autre valeur (absente, `false`) est refusée.

    **Réponse (200) :**
    - `status` : `"ok"` en cas de succès.
    - `message` : message lisible (« Index reconstruit avec succès. »).
    - `chunks_count` : nombre de chunks indexés dans le nouvel index.
    - `last_rebuild_at` : horodatage UTC ISO 8601 de la reconstruction.

    **Codes d'erreur :**
    - `400` : `confirm` absent ou différent de `true`.
    - `500` : échec de la reconstruction (clé Mistral absente, embeddings en
      erreur, etc.).

    Après une reconstruction réussie, le retriever en cache du service RAG est
    invalidé via `RagService.reset_retriever_cache()` afin que le prochain
    `POST /ask` reparte du nouvel index. En production, cet endpoint devrait
    être protégé (auth, rôle dédié) ou remplacé par une tâche planifiée.
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
