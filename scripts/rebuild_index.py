"""Commande principale de reconstruction du vector store FAISS LangChain.

Cette commande reconstruit l'index complet à partir des documents OpenAgenda
préparés : chargement des documents, chunking, génération des embeddings
Mistral, construction du vector store ``langchain_community.vectorstores.FAISS``
et sauvegarde via ``save_local()``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permet d'importer echo_app.* et src.* quand le script est exécuté depuis scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    """Reconstruit le vector store FAISS LangChain à partir des documents préparés."""
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
        embeddings_vectors = embed_texts(chunk_texts)
    except RuntimeError as exc:
        raise SystemExit(
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

    embedding_dimension = (
        len(embeddings_vectors[0]) if embeddings_vectors else 0
    )

    print(f"Nombre de documents chargés : {len(documents)}")
    print(f"Nombre de chunks construits : {len(chunks)}")
    print(f"Nombre d'embeddings générés : {len(embeddings_vectors)}")
    print(f"Dimension de l'index : {embedding_dimension}")
    print(f"Nombre de vecteurs dans l'index : {vectorstore.index.ntotal}")
    print(f"Index sauvegardé : {output_dir / FAISS_INDEX_FILENAME}")
    print(f"Docstore sauvegardé : {output_dir / FAISS_DOCSTORE_FILENAME}")


if __name__ == "__main__":
    rebuild_index()
