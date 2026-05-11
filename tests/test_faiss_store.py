"""Tests unitaires pour le stockage local de l'index FAISS."""

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.indexing.faiss_store import (
    FAISS_INDEX_FILENAME,
    METADATA_FILENAME,
    build_faiss_index,
    build_metadata,
    load_vector_store,
    save_vector_store,
    search_index,
)


def make_chunk(**overrides: object) -> dict:
    """Construit un chunk minimal pour les tests."""
    chunk = {
        "chunk_id": "evt-1_0",
        "event_id": "evt-1",
        "chunk_index": 0,
        "chunk_count": 1,
        "chunk_text": "# Concert\n\nUn concert sur le bassin d'Arcachon.",
        "metadata": {
            "title": "Concert",
            "city": "Arcachon",
            "url": "https://example.com/event",
        },
    }
    chunk.update(overrides)
    return chunk


def test_build_faiss_index_creates_index_with_expected_size() -> None:
    """L'index doit contenir autant de vecteurs que d'embeddings fournis."""
    embeddings = [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
    ]

    index = build_faiss_index(embeddings)

    assert index.ntotal == 2
    assert index.d == 3


def test_build_faiss_index_raises_when_embeddings_are_empty() -> None:
    """Une liste vide d'embeddings doit être rejetée clairement."""
    with pytest.raises(ValueError, match="liste d'embeddings est vide"):
        build_faiss_index([])


def test_build_metadata_keeps_chunk_fields_and_copies_metadata() -> None:
    """Les métadonnées sauvegardées doivent conserver les champs utiles."""
    chunk = make_chunk()

    metadata_entries = build_metadata([chunk])

    assert metadata_entries == [
        {
            "faiss_id": 0,
            "chunk_id": "evt-1_0",
            "event_id": "evt-1",
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_text": "# Concert\n\nUn concert sur le bassin d'Arcachon.",
            "metadata": {
                "title": "Concert",
                "city": "Arcachon",
                "url": "https://example.com/event",
            },
        }
    ]
    assert metadata_entries[0]["metadata"] is not chunk["metadata"]


def test_save_vector_store_creates_expected_files(tmp_path) -> None:
    """La sauvegarde doit créer l'index et le fichier JSON des métadonnées."""
    index = build_faiss_index([[0.0, 1.0], [1.0, 0.0]])
    metadata_entries = build_metadata([make_chunk(), make_chunk(chunk_id="evt-2_0")])

    save_vector_store(index, metadata_entries, output_dir=tmp_path)

    assert (tmp_path / FAISS_INDEX_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()


def test_load_vector_store_reloads_index_and_metadata(tmp_path) -> None:
    """Le rechargement doit restituer l'index FAISS et le mapping JSON."""
    index = build_faiss_index([[0.0, 1.0], [1.0, 0.0]])
    metadata_entries = build_metadata([make_chunk(), make_chunk(chunk_id="evt-2_0")])
    save_vector_store(index, metadata_entries, output_dir=tmp_path)

    loaded_index, loaded_metadata = load_vector_store(tmp_path)

    assert loaded_index.ntotal == 2
    assert loaded_index.d == 2
    assert loaded_metadata == metadata_entries


def test_search_index_returns_coherent_neighbors() -> None:
    """La recherche doit retourner les voisins les plus proches."""
    index = build_faiss_index(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ]
    )

    distances, indices = search_index(index, [0.0, 0.0], top_k=2)

    assert indices == [0, 1]
    assert distances == [0.0, 1.0]


def test_search_index_raises_when_top_k_is_invalid() -> None:
    """top_k doit rester strictement positif."""
    index = build_faiss_index([[0.0, 0.0]])

    with pytest.raises(ValueError, match="strictement positif"):
        search_index(index, [0.0, 0.0], top_k=0)
