"""Tests unitaires pour la construction des documents textuels RAG."""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.documents import (
    build_event_document,
    build_event_document_text,
    build_event_documents,
    get_document_stats,
    save_documents_jsonl,
)


def make_clean_event(**overrides: object) -> dict:
    """Construit un événement nettoyé minimal pour les tests."""
    event = {
        "event_id": "evt-1",
        "title": "Concert de printemps",
        "description": "Un concert en plein air.\n\n- Point 1\n- Point 2",
        "conditions": "Gratuit sur inscription",
        "city": "Arcachon",
        "location_name": "Parc Mauresque",
        "address": "Parc Mauresque, 33120 Arcachon",
        "start_date": "2026-05-10T18:00:00+00:00",
        "end_date": "2026-05-10T20:00:00+00:00",
        "keywords": "musique, plein air",
        "url": "https://example.com/event",
        "image_url": "https://example.com/image.jpg",
        "latitude": 44.84,
        "longitude": -0.58,
        "source": "OpenAgenda",
        "attendance_mode": "offline",
        "status": "scheduled",
    }
    event.update(overrides)
    return event


def test_build_event_document_text_contains_main_fields() -> None:
    """Vérifie que le texte Markdown contient les sections principales."""
    event = make_clean_event()

    document_text = build_event_document_text(event)

    assert document_text.startswith("# Concert de printemps")
    assert "## Description" in document_text
    assert "## Informations pratiques" in document_text
    assert "## Source" in document_text
    assert "Un concert en plein air.\n\n- Point 1\n- Point 2" in document_text
    assert "- Dates : du 2026-05-10 18:00 au 2026-05-10 20:00" in document_text
    assert "- Lieu : Parc Mauresque" in document_text
    assert "- Ville : Arcachon" in document_text
    assert "- Organisateur : OpenAgenda" in document_text
    assert "https://example.com/event" not in document_text


def test_build_event_document_text_skips_empty_fields() -> None:
    """Vérifie que les champs vides ne sont pas ajoutés au texte."""
    event = make_clean_event(
        keywords=None,
        image_url=None,
        attendance_mode=None,
        status=None,
        address=None,
        url=None,
    )

    document_text = build_event_document_text(event)

    assert "None" not in document_text
    assert "- Mots-clés :" not in document_text
    assert "- Mode de participation :" not in document_text
    assert "- Statut :" not in document_text
    assert "- Adresse :" not in document_text
    assert "- URL :" not in document_text


def test_build_event_document_keeps_metadata() -> None:
    """Vérifie que les métadonnées importantes sont conservées."""
    event = make_clean_event()

    document = build_event_document(event)

    assert document is not None
    assert document["event_id"] == "evt-1"
    assert "document_text" in document
    assert "metadata" in document
    assert document["metadata"]["title"] == "Concert de printemps"
    assert document["metadata"]["city"] == "Arcachon"
    assert document["metadata"]["url"] == "https://example.com/event"
    assert document["metadata"]["source"] == "OpenAgenda"


def test_build_event_documents_ignores_events_without_id() -> None:
    """Vérifie que les événements sans identifiant sont ignorés."""
    events = [
        make_clean_event(event_id="evt-1"),
        make_clean_event(event_id=None, title="Sans identifiant"),
    ]

    documents = build_event_documents(events)

    assert len(documents) == 1
    assert documents[0]["event_id"] == "evt-1"


def test_save_documents_jsonl_writes_one_document_per_line(tmp_path) -> None:
    """Vérifie que la sauvegarde JSONL écrit un document par ligne."""
    documents = [
        {"event_id": "evt-1", "document_text": "# A", "metadata": {"title": "A"}},
        {"event_id": "evt-2", "document_text": "# B", "metadata": {"title": "B"}},
    ]
    output_path = tmp_path / "documents.jsonl"

    saved_path = save_documents_jsonl(documents, output_path)

    assert saved_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_id"] == "evt-1"
    assert json.loads(lines[1])["event_id"] == "evt-2"


def test_get_document_stats() -> None:
    """Vérifie le calcul des statistiques sur les documents."""
    documents = [
        {"event_id": "evt-1", "document_text": "# Titre\n\n## Description\nTexte", "metadata": {}},
        {"event_id": "evt-2", "document_text": "", "metadata": {}},
    ]

    stats = get_document_stats(documents)

    assert stats["total_documents"] == 2
    assert stats["min_text_length"] == 0
    assert stats["max_text_length"] == len("# Titre\n\n## Description\nTexte")
    assert stats["empty_text_count"] == 1
    assert stats["mean_text_length"] == 14.5
