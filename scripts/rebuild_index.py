"""Commande principale de reconstruction du vector store FAISS local.

Cette commande reconstruit l'index complet à partir des documents OpenAgenda
préparés : chargement des documents, chunking, génération des embeddings
Mistral, construction de l'index FAISS et sauvegarde des métadonnées.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permet d'importer echo_app.* et src.* quand le script est exécuté depuis scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.config import get_mistral_api_key
from echo_app.indexing.embeddings import embed_texts
from echo_app.indexing.faiss_store import (
    FAISS_INDEX_FILENAME,
    METADATA_FILENAME,
    VECTOR_STORE_DIR,
    build_faiss_index,
    build_metadata,
    save_vector_store,
)
from src.chunking import build_chunks
from src.config import PATHS, PROCESSED_EVENTS_DOCUMENTS_FILENAME


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


def rebuild_index() -> None:
    """Reconstruit le vector store local complet à partir des documents préparés."""
    input_path = PATHS.data_processed / PROCESSED_EVENTS_DOCUMENTS_FILENAME

    try:
        get_mistral_api_key()
    except ValueError as exc:
        raise SystemExit(
            "MISTRAL_API_KEY est absente. Configurez-la dans .env avant de construire l'index."
        ) from exc

    documents = load_documents(input_path)
    chunks = build_chunks(documents)
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]

    try:
        embeddings = embed_texts(chunk_texts)
    except RuntimeError as exc:
        raise SystemExit(
            "La génération des embeddings Mistral a échoué. "
            "Vérifiez la connectivité réseau et la configuration Mistral."
        ) from exc

    index = build_faiss_index(embeddings)
    metadata = build_metadata(chunks)
    save_vector_store(index, metadata, output_dir=VECTOR_STORE_DIR)

    print(f"Nombre de documents chargés : {len(documents)}")
    print(f"Nombre de chunks construits : {len(chunks)}")
    print(f"Nombre d'embeddings générés : {len(embeddings)}")
    print(f"Dimension de l'index : {index.d}")
    print(f"Nombre de vecteurs dans l'index : {index.ntotal}")
    print(f"Index sauvegardé : {VECTOR_STORE_DIR / FAISS_INDEX_FILENAME}")
    print(f"Métadonnées sauvegardées : {VECTOR_STORE_DIR / METADATA_FILENAME}")


if __name__ == "__main__":
    rebuild_index()
