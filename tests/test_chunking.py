"""Tests unitaires pour le chunking des documents OpenAgenda."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chunking import build_chunks, chunk_long_text, get_chunk_stats


def make_document(**overrides: object) -> dict:
    """Construit un document RAG minimal pour les tests."""
    document = {
        "event_id": "evt-1",
        "document_text": "# Concert de printemps\n\nUn concert en plein air.",
        "metadata": {
            "title": "Concert de printemps",
            "city": "Arcachon",
            "url": "https://example.com/event",
        },
    }
    document.update(overrides)
    return document


def make_long_document() -> dict:
    """Construit un document long qui doit être découpé."""
    repeated_paragraph = (
        "Le festival propose des concerts, des ateliers et des rencontres "
        "dans plusieurs lieux du bassin d'Arcachon. "
    )
    long_text = "# Grand festival\n\n" + (repeated_paragraph * 20)

    return make_document(
        event_id="evt-long",
        document_text=long_text,
        metadata={
            "title": "Grand festival",
            "city": "La Teste-de-Buch",
            "url": "https://example.com/festival",
        },
    )


def test_chunk_long_text_keeps_short_text_in_one_chunk() -> None:
    """Un texte court doit rester dans un seul chunk."""
    text = "Texte court et suffisant pour rester entier."

    chunks = chunk_long_text(text)

    assert chunks == [text]


def test_chunk_long_text_splits_long_text() -> None:
    """Un texte long doit produire plusieurs chunks."""
    text = ("Paragraphe de test " * 60).strip()

    chunks = chunk_long_text(text, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) >= 20 for chunk in chunks)


def test_build_chunks_keeps_event_id_and_metadata() -> None:
    """Les chunks doivent conserver l'identifiant source et les métadonnées."""
    document = make_long_document()

    chunks = build_chunks([document], chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk["event_id"] == "evt-long" for chunk in chunks)
    assert all(chunk["metadata"]["city"] == "La Teste-de-Buch" for chunk in chunks)
    assert chunks[0]["metadata"] == document["metadata"]
    assert chunks[0]["metadata"] is not document["metadata"]


def test_build_chunks_uses_stable_chunk_ids() -> None:
    """Les identifiants de chunks doivent être stables et lisibles."""
    document = make_long_document()

    first_run = build_chunks([document], chunk_size=120, chunk_overlap=20)
    second_run = build_chunks([document], chunk_size=120, chunk_overlap=20)

    assert [chunk["chunk_id"] for chunk in first_run] == [
        chunk["chunk_id"] for chunk in second_run
    ]
    assert first_run[0]["chunk_id"] == "evt-long_0"
    assert first_run[1]["chunk_id"] == "evt-long_1"


def test_build_chunks_ignores_invalid_documents() -> None:
    """Les documents sans identifiant ou sans texte exploitable sont ignorés."""
    documents = [
        make_document(event_id=None),
        make_document(event_id="evt-2", document_text="   "),
        make_document(event_id="evt-3"),
    ]

    chunks = build_chunks(documents)

    assert len(chunks) == 1
    assert chunks[0]["event_id"] == "evt-3"


def test_build_chunks_adds_title_when_document_is_split() -> None:
    """Le titre doit rester visible sur les chunks d'un document découpé."""
    document = make_long_document()

    chunks = build_chunks([document], chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk["chunk_text"].startswith("# Grand festival") for chunk in chunks)


def test_build_chunks_never_returns_empty_chunks() -> None:
    """Le chunking ne doit pas produire de chunk vide."""
    document = make_document(
        event_id="evt-spaces",
        document_text="# Atelier\n\n" + ("Texte utile   \n\n" * 40),
    )

    chunks = build_chunks([document], chunk_size=100, chunk_overlap=15)

    assert chunks
    assert all(chunk["chunk_text"].strip() for chunk in chunks)


def test_get_chunk_stats_returns_simple_stats() -> None:
    """Les statistiques de chunking doivent rester simples et lisibles."""
    documents = [
        make_document(event_id="evt-short"),
        make_long_document(),
    ]
    chunks = build_chunks(documents, chunk_size=120, chunk_overlap=20)

    stats = get_chunk_stats(chunks)

    assert stats["total_chunks"] == len(chunks)
    assert stats["total_events"] == 2
    assert stats["single_chunk_events"] == 1
    assert stats["multi_chunk_events"] == 1
    assert stats["max_chunk_length"] >= stats["min_chunk_length"]
    assert stats["empty_chunk_count"] == 0
