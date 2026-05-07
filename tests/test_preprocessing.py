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
)


def test_clean_html_removes_tags():
    assert clean_html("<p>Hello <strong>world</strong></p>") == "Hello world"


def test_build_clean_description_avoids_duplicate_short_description():
    event = {
        "description_fr": "Résumé court",
        "longdescription_fr": "<p>Résumé court avec plus de détails.</p>",
        "conditions_fr": None,
    }

    description = build_clean_description(event)

    assert description == "Résumé court avec plus de détails."
    assert description.count("Résumé court") == 1


def test_build_clean_description_concatenates_when_descriptions_are_different():
    event = {
        "description_fr": "Résumé court",
        "longdescription_fr": "<p>Description longue différente.</p>",
        "conditions_fr": "Gratuit",
    }

    description = build_clean_description(event)

    assert "Résumé court" in description
    assert "Description longue différente" in description
    assert "Conditions : Gratuit" in description


def test_format_keywords_removes_technical_challenge_ids():
    keywords = ["Nature", "challenge-id=123", "Astronomie"]

    assert format_keywords(keywords) == "Nature, Astronomie"


def test_extract_coordinates_from_dict():
    event = {"location_coordinates": {"lat": 44.1, "lon": -1.2}}

    assert extract_coordinates(event) == (44.1, -1.2)


def test_filter_events_by_date_keeps_current_events():
    reference_datetime = datetime(2026, 5, 1, tzinfo=timezone.utc)
    events = [
        {"uid": "1", "lastdate_end": "2026-05-10T10:00:00+00:00"},
        {"uid": "2", "lastdate_end": "2026-04-10T10:00:00+00:00"},
    ]

    filtered_events = filter_events_by_date(events, reference_datetime)

    assert len(filtered_events) == 1
    assert filtered_events[0]["uid"] == "1"


def test_clean_event_returns_expected_fields():
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
