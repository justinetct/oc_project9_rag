"""Reconstruction du vector store FAISS LangChain — fonction applicative.

Ce module contient la logique de reconstruction de l'index, sous forme
de fonction réutilisable. Elle peut être appelée :

- depuis la ligne de commande via ``scripts/rebuild_index.py`` ;
- depuis l'API FastAPI via ``POST /rebuild``.

La fonction écrit aussi un petit fichier ``rebuild_metadata.json`` à
côté de l'index. Ce fichier est généré localement et n'est pas versionné
(``vector_store/`` est dans ``.gitignore``). Il sert à exposer
``chunks_count`` et ``last_rebuild_at`` dans ``GET /health`` sans avoir
à recharger le vector store complet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from echo_app.config import get_mistral_api_key
from echo_app.indexing.embeddings import embed_texts
from echo_app.indexing.langchain_embeddings import MistralLangChainEmbeddings
from echo_app.indexing.langchain_faiss_store import (
    FAISS_DOCSTORE_FILENAME,
    FAISS_INDEX_FILENAME,
    VECTOR_STORE_DIR,
    build_langchain_vector_store_from_embeddings,
    chunks_to_documents,
    save_langchain_vector_store,
)
from src.chunking import build_chunks
from src.config import PATHS, PROCESSED_EVENTS_DOCUMENTS_FILENAME


REBUILD_METADATA_FILENAME = "rebuild_metadata.json"


def _utc_now_iso() -> str:
    """Retourne l'instant courant au format ISO 8601 UTC (``...Z``)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_documents(path: Path) -> list[dict]:
    """Charge un fichier JSONL de documents OpenAgenda."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    documents: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            documents.append(json.loads(stripped_line))

    return documents


def _write_rebuild_metadata(
    output_dir: Path,
    chunks_count: int,
    last_rebuild_at: str,
) -> Path:
    """Écrit ``rebuild_metadata.json`` à côté de l'index FAISS."""
    metadata_path = output_dir / REBUILD_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(
            {
                "chunks_count": chunks_count,
                "last_rebuild_at": last_rebuild_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata_path


def read_rebuild_metadata(
    vector_store_dir: Path = VECTOR_STORE_DIR,
) -> dict | None:
    """Lit ``rebuild_metadata.json`` si présent, sinon retourne ``None``.

    Le chemin est résolu de la même façon que celui du vector store :
    relatif à la racine du projet sauf si absolu.
    """
    if not vector_store_dir.is_absolute():
        vector_store_dir = PATHS.root / vector_store_dir
    metadata_path = vector_store_dir / REBUILD_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def rebuild_index() -> dict:
    """Reconstruit le vector store FAISS LangChain à partir des documents préparés.

    Retourne un dictionnaire de statistiques utilisable depuis l'API ou
    depuis le CLI. Lève ``RuntimeError`` en cas d'erreur de configuration
    ou d'appel Mistral (afin que l'API puisse l'attraper proprement).
    """
    input_path = PATHS.data_processed / PROCESSED_EVENTS_DOCUMENTS_FILENAME

    try:
        get_mistral_api_key()
    except ValueError as exc:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente. Configurez-la dans .env avant de "
            "reconstruire l'index."
        ) from exc

    documents = load_documents(input_path)
    chunks = build_chunks(documents)
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]

    try:
        embeddings_vectors = embed_texts(chunk_texts)
    except RuntimeError as exc:
        raise RuntimeError(
            "La génération des embeddings Mistral a échoué. "
            "Vérifiez la connectivité réseau et la configuration Mistral."
        ) from exc

    langchain_documents = chunks_to_documents(chunks)
    metadatas = [doc.metadata for doc in langchain_documents]

    embeddings_client = MistralLangChainEmbeddings()
    vectorstore = build_langchain_vector_store_from_embeddings(
        chunk_texts=chunk_texts,
        chunk_embeddings=embeddings_vectors,
        metadatas=metadatas,
        embeddings=embeddings_client,
    )

    output_dir = save_langchain_vector_store(vectorstore, output_dir=VECTOR_STORE_DIR)

    last_rebuild_at = _utc_now_iso()
    _write_rebuild_metadata(
        output_dir=output_dir,
        chunks_count=len(chunks),
        last_rebuild_at=last_rebuild_at,
    )

    embedding_dimension = len(embeddings_vectors[0]) if embeddings_vectors else 0

    return {
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "embeddings_count": len(embeddings_vectors),
        "embedding_dimension": embedding_dimension,
        "vectors_count": vectorstore.index.ntotal,
        "vector_store_path": output_dir,
        "index_file": output_dir / FAISS_INDEX_FILENAME,
        "docstore_file": output_dir / FAISS_DOCSTORE_FILENAME,
        "last_rebuild_at": last_rebuild_at,
    }
