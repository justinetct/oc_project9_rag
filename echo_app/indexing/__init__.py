"""Outils d'indexation pour l'application Écho."""

from echo_app.indexing.embeddings import embed_query, embed_texts, get_mistral_client
from echo_app.indexing.faiss_store import (
    build_faiss_index,
    build_metadata,
    load_vector_store,
    save_vector_store,
    search_index,
)
from echo_app.indexing.langchain_embeddings import MistralLangChainEmbeddings
from echo_app.indexing.langchain_faiss_store import (
    build_langchain_vector_store,
    build_langchain_vector_store_from_embeddings,
    chunks_to_documents,
    load_langchain_vector_store,
    save_langchain_vector_store,
)
from echo_app.indexing.search import search_similar_events

__all__ = [
    "MistralLangChainEmbeddings",
    "build_faiss_index",
    "build_langchain_vector_store",
    "build_langchain_vector_store_from_embeddings",
    "build_metadata",
    "chunks_to_documents",
    "embed_query",
    "embed_texts",
    "get_mistral_client",
    "load_langchain_vector_store",
    "load_vector_store",
    "save_langchain_vector_store",
    "save_vector_store",
    "search_index",
    "search_similar_events",
]
