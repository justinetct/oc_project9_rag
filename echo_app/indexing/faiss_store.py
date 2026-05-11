"""Fonctions simples pour construire et sauvegarder un index FAISS local."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import faiss
import numpy as np

from src.config import PATHS


VECTOR_STORE_DIR = Path("vector_store")
FAISS_INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


def _resolve_store_dir(path: Path) -> Path:
    """Résout le dossier du vector store depuis la racine du projet."""
    if path.is_absolute():
        return path
    return PATHS.root / path


def build_faiss_index(embeddings: list[list[float]]) -> faiss.Index:
    """Construit un index FAISS plat à partir d'une liste d'embeddings."""
    if not embeddings:
        raise ValueError("La liste d'embeddings est vide.")

    dimension = len(embeddings[0])
    if dimension == 0:
        raise ValueError("Les embeddings doivent avoir une dimension non nulle.")

    if any(len(embedding) != dimension for embedding in embeddings):
        raise ValueError("Les embeddings n'ont pas tous la même dimension.")

    embeddings_array = np.asarray(embeddings, dtype="float32")
    if embeddings_array.ndim != 2:
        raise ValueError("Les embeddings doivent former un tableau 2D.")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index


def build_metadata(chunks: list[dict]) -> list[dict]:
    """Construit le mapping entre les positions FAISS et les chunks source."""
    metadata_entries: list[dict] = []

    for faiss_id, chunk in enumerate(chunks):
        metadata_entries.append(
            {
                "faiss_id": faiss_id,
                "chunk_id": chunk.get("chunk_id"),
                "event_id": chunk.get("event_id"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_count": chunk.get("chunk_count"),
                "chunk_text": chunk.get("chunk_text"),
                "metadata": deepcopy(chunk.get("metadata", {})),
            }
        )

    return metadata_entries


def save_vector_store(
    index: faiss.Index,
    metadata: list[dict],
    output_dir: Path = VECTOR_STORE_DIR,
) -> None:
    """Sauvegarde l'index FAISS et les métadonnées associées en local."""
    output_dir = _resolve_store_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / FAISS_INDEX_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_vector_store(
    input_dir: Path = VECTOR_STORE_DIR,
) -> tuple[faiss.Index, list[dict]]:
    """Recharge un index FAISS local et son mapping de métadonnées."""
    input_dir = _resolve_store_dir(input_dir)
    index_path = input_dir / FAISS_INDEX_FILENAME
    metadata_path = input_dir / METADATA_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")

    index = faiss.read_index(str(index_path))

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search_index(
    index: faiss.Index,
    query_embedding: list[float],
    top_k: int = 5,
) -> tuple[list[float], list[int]]:
    """Recherche les vecteurs les plus proches d'un embedding de requête."""
    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")

    query_array = np.asarray(query_embedding, dtype="float32")
    if query_array.ndim != 1:
        raise ValueError("query_embedding doit être un vecteur 1D.")
    if query_array.size != index.d:
        raise ValueError("La dimension du vecteur de requête est incohérente.")

    distances, indices = index.search(query_array.reshape(1, -1), top_k)

    return distances[0].tolist(), indices[0].tolist()
