"""Fonctions simples de filtrage temporel des événements OpenAgenda."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def parse_event_datetime(value: str | None) -> datetime | None:
    """Convertit une date OpenAgenda en datetime timezone-aware."""
    if not value:
        return None

    normalized_value = value.strip()
    if not normalized_value:
        return None

    if normalized_value.endswith("Z"):
        normalized_value = normalized_value[:-1] + "+00:00"

    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None

    if parsed_value.tzinfo is None:
        parsed_value = parsed_value.replace(tzinfo=timezone.utc)

    return parsed_value


def get_event_reference_end_datetime(event: dict) -> datetime | None:
    """Retourne la meilleure date de fin exploitable d'un événement."""
    for field_name in ["lastdate_end", "firstdate_end", "firstdate_begin"]:
        parsed_value = parse_event_datetime(event.get(field_name))
        if parsed_value is not None:
            return parsed_value
    return None


def filter_events_by_date(
    events: list[dict],
    reference_datetime: datetime,
) -> list[dict]:
    """Conserve les événements encore actifs ou à venir."""
    filtered_events: list[dict] = []

    for event in events:
        event_datetime = get_event_reference_end_datetime(event)
        if event_datetime is None:
            continue

        if event_datetime >= reference_datetime:
            filtered_events.append(event)

    return filtered_events


def get_date_filtering_stats(
    events: list[dict],
    filtered_events: list[dict],
) -> dict:
    """Calcule quelques statistiques simples sur le filtrage."""
    total_events = len(events)
    kept_events = len(filtered_events)
    excluded_events = total_events - kept_events
    excluded_ratio = round(excluded_events / total_events, 3) if total_events else 0.0

    return {
        "total_events": total_events,
        "kept_events": kept_events,
        "excluded_events": excluded_events,
        "excluded_ratio": excluded_ratio,
    }


def save_filtered_events(events: list[dict], output_path: Path) -> Path:
    """Sauvegarde la liste filtrée d'événements au format JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    return output_path
