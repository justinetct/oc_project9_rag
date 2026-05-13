"""Tests unitaires pour le vector store FAISS LangChain.

Les embeddings sont remplacés par un fake déterministe (FakeEmbeddings)
pour éviter tout appel réseau à Mistral. Les fichiers
``index.faiss`` / ``index.pkl`` sont écrits et relus dans ``tmp_path``.
"""

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from echo_app.indexing.langchain_faiss_store import (  # noqa: E402
    LEGACY_METADATA_FILENAME,
    build_langchain_vector_store,
    build_langchain_vector_store_from_embeddings,
    chunks_to_documents,
    load_langchain_vector_store,
    save_langchain_vector_store,
)


class FakeEmbeddings(Embeddings):
    """Embeddings déterministes pour les tests : un vecteur 4D simple par texte.

    On évite tout appel réseau et on garantit que des textes différents
    produisent des vecteurs différents pour que la recherche FAISS soit
    discriminante.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _vectorize(self, text: str) -> list[float]:
        normalized = (text or "").strip().lower()
        if "astronomie" in normalized:
            return [1.0, 0.0, 0.0, 0.0]
        if "concert" in normalized:
            return [0.0, 1.0, 0.0, 0.0]
        if "exposition" in normalized:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def make_chunk(chunk_id: str, event_id: str, text: str, **metadata) -> dict:
    """Construit un chunk type pour alimenter les tests."""
    return {
        "chunk_id": chunk_id,
        "event_id": event_id,
        "chunk_index": 0,
        "chunk_count": 1,
        "chunk_text": text,
        "metadata": metadata,
    }


def test_chunks_to_documents_maps_text_and_metadata() -> None:
    """chunks_to_documents doit produire des Document avec metadata à plat."""
    chunks = [
        make_chunk(
            chunk_id="evt-1_0",
            event_id="evt-1",
            text="Soirée astronomie à Lanton.",
            title="Initiation à l'astronomie",
            city="Lanton",
            start_date="2026-01-05T19:30",
            url="https://example.com/evt-1",
        )
    ]

    documents = chunks_to_documents(chunks)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert document.page_content == "Soirée astronomie à Lanton."
    assert document.metadata["event_id"] == "evt-1"
    assert document.metadata["chunk_id"] == "evt-1_0"
    assert document.metadata["chunk_index"] == 0
    assert document.metadata["chunk_count"] == 1
    assert document.metadata["title"] == "Initiation à l'astronomie"
    assert document.metadata["city"] == "Lanton"
    assert document.metadata["start_date"] == "2026-01-05T19:30"
    assert document.metadata["url"] == "https://example.com/evt-1"


def test_chunks_to_documents_handles_missing_metadata() -> None:
    """Un chunk sans metadata ne doit pas faire planter la conversion."""
    chunks = [
        {
            "chunk_id": "evt-2_0",
            "event_id": "evt-2",
            "chunk_text": "Texte",
        }
    ]

    documents = chunks_to_documents(chunks)

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == "Texte"
    assert document.metadata["event_id"] == "evt-2"
    assert document.metadata["chunk_id"] == "evt-2_0"
    assert document.metadata.get("title") is None
    assert document.metadata.get("city") is None


def test_build_save_load_round_trip(tmp_path) -> None:
    """Construire, sauvegarder, recharger, puis interroger via .as_retriever()."""
    documents = chunks_to_documents(
        [
            make_chunk(
                chunk_id="evt-1_0",
                event_id="evt-1",
                text="Soirée astronomie à Lanton.",
                title="Astronomie",
                city="Lanton",
            ),
            make_chunk(
                chunk_id="evt-2_0",
                event_id="evt-2",
                text="Concert acoustique à Andernos.",
                title="Concert",
                city="Andernos",
            ),
        ]
    )

    embeddings = FakeEmbeddings()
    vectorstore = build_langchain_vector_store(documents, embeddings)
    save_langchain_vector_store(vectorstore, output_dir=tmp_path)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "index.pkl").exists()

    reloaded = load_langchain_vector_store(embeddings, input_dir=tmp_path)
    retriever = reloaded.as_retriever(search_kwargs={"k": 1})
    hits = retriever.invoke("concert")

    assert len(hits) == 1
    assert isinstance(hits[0], Document)
    assert hits[0].metadata["event_id"] == "evt-2"


def test_save_removes_legacy_metadata_json(tmp_path) -> None:
    """save_langchain_vector_store doit supprimer l'ancien metadata.json."""
    legacy_path = tmp_path / LEGACY_METADATA_FILENAME
    legacy_path.write_text("[]", encoding="utf-8")

    documents = chunks_to_documents(
        [make_chunk("evt-1_0", "evt-1", "texte", title="t", city="c")]
    )
    vectorstore = build_langchain_vector_store(documents, FakeEmbeddings())
    save_langchain_vector_store(vectorstore, output_dir=tmp_path)

    assert not legacy_path.exists()
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "index.pkl").exists()


def test_load_raises_when_index_missing(tmp_path) -> None:
    """load_langchain_vector_store doit lever FileNotFoundError si absent."""
    with pytest.raises(FileNotFoundError, match="Index FAISS LangChain"):
        load_langchain_vector_store(FakeEmbeddings(), input_dir=tmp_path)


def test_build_from_embeddings_preserves_metadata(tmp_path) -> None:
    """build_langchain_vector_store_from_embeddings doit conserver les metadata."""
    embeddings = FakeEmbeddings()
    chunk_texts = ["Soirée astronomie à Lanton.", "Concert à Andernos."]
    chunk_embeddings = [
        embeddings.embed_query(chunk_texts[0]),
        embeddings.embed_query(chunk_texts[1]),
    ]
    metadatas = [
        {"event_id": "evt-1", "chunk_id": "evt-1_0", "title": "Astro"},
        {"event_id": "evt-2", "chunk_id": "evt-2_0", "title": "Concert"},
    ]

    # On teste le chemin utilisé par rebuild_index.py : les embeddings sont
    # déjà calculés, puis associés aux textes et métadonnées dans FAISS.
    vectorstore = build_langchain_vector_store_from_embeddings(
        chunk_texts=chunk_texts,
        chunk_embeddings=chunk_embeddings,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    # On vérifie aussi le format persistant LangChain : index.faiss + index.pkl.
    save_langchain_vector_store(vectorstore, output_dir=tmp_path)
    reloaded = load_langchain_vector_store(embeddings, input_dir=tmp_path)

    # Le retriever doit retrouver le bon document avec ses métadonnées.
    retriever = reloaded.as_retriever(search_kwargs={"k": 1})
    hits = retriever.invoke("astronomie")

    assert len(hits) == 1
    assert hits[0].metadata["event_id"] == "evt-1"
    assert hits[0].metadata["title"] == "Astro"


def test_build_from_embeddings_validates_length() -> None:
    """Des longueurs incohérentes entre textes/embeddings/metadatas doivent échouer."""
    with pytest.raises(ValueError, match="même longueur"):
        build_langchain_vector_store_from_embeddings(
            chunk_texts=["a", "b"],
            chunk_embeddings=[[0.1]],
            metadatas=[{}, {}],
            embeddings=FakeEmbeddings(),
        )


def test_build_rejects_empty_documents() -> None:
    """Une liste de documents vide doit lever une erreur claire."""
    with pytest.raises(ValueError, match="vide"):
        build_langchain_vector_store([], FakeEmbeddings())
