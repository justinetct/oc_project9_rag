"""Vector store FAISS LangChain pour le pipeline RAG d'Écho.

Ce module fournit les utilitaires pour :

- convertir les chunks du projet en objets ``Document`` LangChain ;
- construire un vector store ``langchain_community.vectorstores.FAISS`` ;
- sauvegarder localement l'index avec ``save_local()`` ;
- recharger l'index avec ``FAISS.load_local()``.

L'objectif est de remplacer l'index FAISS maison par le vector store
standard de LangChain. Le format actif sur disque est désormais
``index.faiss`` + ``index.pkl`` (créés par ``save_local()``).
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import PATHS


VECTOR_STORE_DIR = Path("vector_store")
FAISS_INDEX_FILENAME = "index.faiss"
FAISS_DOCSTORE_FILENAME = "index.pkl"
LEGACY_METADATA_FILENAME = "metadata.json"


def _resolve_store_dir(path: Path) -> Path:
    """Résout le dossier du vector store depuis la racine du projet."""
    if path.is_absolute():
        return path
    return PATHS.root / path


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """Convertit les chunks du projet en Document LangChain.

    Les métadonnées événement (title, city, start_date, url, etc.) sont
    placées à plat dans ``metadata`` aux côtés de ``event_id`` et
    ``chunk_id``. Cela permet aux helpers de ``RagService`` de lire
    directement les champs sans navigation imbriquée.
    """
    documents: list[Document] = []
    for chunk in chunks:
        base_metadata = chunk.get("metadata") or {}
        documents.append(
            Document(
                page_content=chunk.get("chunk_text") or "",
                metadata={
                    **base_metadata,
                    "event_id": chunk.get("event_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "chunk_count": chunk.get("chunk_count"),
                },
            )
        )
    return documents


def build_langchain_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
) -> FAISS:
    """Construit un vector store FAISS LangChain depuis des Documents.

    Délègue à ``FAISS.from_documents`` ; les embeddings sont calculés via
    le client ``embeddings`` fourni (typiquement
    ``MistralLangChainEmbeddings``).
    """
    if not documents:
        raise ValueError("La liste de documents est vide.")
    return FAISS.from_documents(documents=documents, embedding=embeddings)


def build_langchain_vector_store_from_embeddings(
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    metadatas: list[dict],
    embeddings: Embeddings,
) -> FAISS:
    """Construit un vector store FAISS LangChain à partir d'embeddings déjà calculés.

    Utile depuis ``rebuild_index.py`` pour ne pas re-déclencher les appels
    Mistral après avoir déjà appelé ``embed_texts(chunk_texts)``.
    """
    if not chunk_texts:
        raise ValueError("La liste de chunks est vide.")
    if len(chunk_texts) != len(chunk_embeddings):
        raise ValueError(
            "chunk_texts et chunk_embeddings doivent avoir la même longueur."
        )
    if len(chunk_texts) != len(metadatas):
        raise ValueError(
            "chunk_texts et metadatas doivent avoir la même longueur."
        )

    text_embedding_pairs = list(zip(chunk_texts, chunk_embeddings, strict=True))
    return FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings,
        metadatas=metadatas,
    )


def save_langchain_vector_store(
    vectorstore: FAISS,
    output_dir: Path = VECTOR_STORE_DIR,
) -> Path:
    """Sauvegarde le vector store FAISS LangChain en local.

    Crée le dossier si nécessaire, supprime tout ancien ``metadata.json``
    résiduel du store maison pour éviter l'ambiguïté entre les deux
    formats, puis appelle ``vectorstore.save_local()``.
    """
    output_dir = _resolve_store_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_metadata = output_dir / LEGACY_METADATA_FILENAME
    if legacy_metadata.exists():
        legacy_metadata.unlink()

    vectorstore.save_local(str(output_dir))
    return output_dir


def load_langchain_vector_store(
    embeddings: Embeddings,
    input_dir: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Recharge un vector store FAISS LangChain depuis le disque."""
    input_dir = _resolve_store_dir(input_dir)
    index_path = input_dir / FAISS_INDEX_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index FAISS LangChain introuvable dans {input_dir}. "
            "Lancer `poetry run python scripts/rebuild_index.py`."
        )

    return FAISS.load_local(
        str(input_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
