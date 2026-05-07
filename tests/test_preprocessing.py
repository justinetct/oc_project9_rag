"""Tests unitaires pour le filtrage et le nettoyage des événements OpenAgenda."""

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    build_clean_description,
    clean_event,
    clean_html,
    extract_coordinates,
    filter_events_by_date,
    format_keywords,
    html_to_markdown_text,
)


def test_clean_html_removes_tags() -> None:
    """Vérifie que les balises HTML sont supprimées du texte."""
    assert clean_html("<p>Hello <strong>world</strong></p>") == "Hello world"


def test_html_to_markdown_text_preserves_paragraphs() -> None:
    """Vérifie que deux paragraphes restent séparés."""
    text = "<p>Premier paragraphe.</p><p>Deuxième paragraphe.</p>"

    result = html_to_markdown_text(text)

    assert result == "Premier paragraphe.\n\nDeuxième paragraphe."


def test_html_to_markdown_text_converts_list_items() -> None:
    """Vérifie que les listes HTML deviennent des puces Markdown."""
    text = "<ul><li>Premier</li><li>Deuxième</li></ul>"

    result = html_to_markdown_text(text)

    assert result == "- Premier\n- Deuxième"


def test_html_to_markdown_text_removes_links_but_keeps_text() -> None:
    """Vérifie que le texte d'un lien est conservé sans son URL."""
    text = (
        '<p>Réservation sur '
        '<a href="https://example.com">EcoNature</a> '
        '<a href="https://example.com">https://example.com</a></p>'
    )

    result = html_to_markdown_text(text)

    assert "EcoNature" in result
    assert "https://example.com" not in result


def test_html_to_markdown_text_removes_repeated_raw_link() -> None:
    """Vérifie qu'un libellé suivi d'une URL brute est simplifié proprement."""
    html_text = (
        '<p>Infos et réservation sur '
        '<a href="https://www.eco-nature.org/experience/demo">EcoNature : </a>'
        '<a href="https://www.eco-nature.org/experience/demo">'
        'https://www.eco-nature.org/experience/demo</a></p>'
    )

    result = html_to_markdown_text(html_text)

    assert "Infos et réservation sur EcoNature" in result
    assert "EcoNature :" not in result
    assert "https://www.eco-nature.org" not in result


def test_build_clean_description_avoids_duplicate_short_description() -> None:
    """Vérifie que la description courte n'est pas dupliquée."""
    event = {
        "description_fr": "Résumé court",
        "longdescription_fr": "<p>Résumé court avec plus de détails.</p>",
        "conditions_fr": None,
    }

    description = build_clean_description(event)

    assert description == "Résumé court avec plus de détails."
    assert description.count("Résumé court") == 1


def test_build_clean_description_concatenates_when_descriptions_are_different() -> None:
    """Vérifie que les descriptions différentes sont concaténées."""
    event = {
        "description_fr": "Résumé court",
        "longdescription_fr": "<p>Description longue différente.</p><ul><li>Point 1</li></ul>",
        "conditions_fr": "Gratuit",
    }

    description = build_clean_description(event)

    assert description.startswith("Résumé court\n\nDescription longue différente.")
    assert "- Point 1" in description
    assert "\n\nConditions : Gratuit" in description


def test_format_keywords_removes_technical_challenge_ids() -> None:
    """Vérifie que les mots-clés techniques sont supprimés."""
    keywords = ["Nature", "challenge-id=123", "Astronomie"]

    assert format_keywords(keywords) == "Nature, Astronomie"


def test_extract_coordinates_from_dict() -> None:
    """Vérifie l'extraction latitude / longitude depuis OpenAgenda."""
    event = {"location_coordinates": {"lat": 44.1, "lon": -1.2}}

    assert extract_coordinates(event) == (44.1, -1.2)


def test_filter_events_by_date_keeps_current_events() -> None:
    """Vérifie que le filtrage conserve seulement les événements actifs."""
    reference_datetime = datetime(2026, 5, 1, tzinfo=timezone.utc)
    events = [
        {"uid": "1", "lastdate_end": "2026-05-10T10:00:00+00:00"},
        {"uid": "2", "lastdate_end": "2026-04-10T10:00:00+00:00"},
    ]

    filtered_events = filter_events_by_date(events, reference_datetime)

    assert len(filtered_events) == 1
    assert filtered_events[0]["uid"] == "1"


def test_clean_event_returns_expected_fields() -> None:
    """Vérifie que le nettoyage retourne les champs attendus."""
    raw_event = {
        "uid": 123,
        "title_fr": "Observation du ciel",
        "description_fr": "Une soirée découverte",
        "conditions_fr": "Gratuit",
        "keywords_fr": ["Nature", "Astronomie"],
        "location_city": "Lanton",
        "location_name": "Parc du ciel",
        "location_address": "1 rue des étoiles",
        "firstdate_begin": "2026-05-10T18:00:00+00:00",
        "firstdate_end": "2026-05-10T20:00:00+00:00",
        "lastdate_end": "2026-05-10T20:00:00+00:00",
        "canonicalurl": "https://example.com/event",
        "location_coordinates": {"lat": 44.7, "lon": -1.0},
        "originagenda_title": "OpenAgenda Bassin",
        "attendancemode": '{"label": {"fr": "Sur place"}}',
        "status": '{"label": {"fr": "Programmé"}}',
        "image": "https://example.com/image.jpg",
    }

    cleaned_event = clean_event(raw_event)

    expected_fields = {
        "event_id",
        "title",
        "description",
        "conditions",
        "city",
        "location_name",
        "address",
        "start_date",
        "end_date",
        "keywords",
        "url",
        "image_url",
        "latitude",
        "longitude",
        "source",
        "attendance_mode",
        "status",
    }

    assert cleaned_event is not None
    assert expected_fields.issubset(cleaned_event.keys())
    assert "age_min" not in cleaned_event
    assert "age_max" not in cleaned_event
    assert "registration_url" not in cleaned_event
    assert "external_url" not in cleaned_event
