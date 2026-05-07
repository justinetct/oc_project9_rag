"""Fonctions simples de filtrage et de nettoyage des événements OpenAgenda."""

from __future__ import annotations

import html
import json
import re
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


def clean_html(text: str | None) -> str:
    """Supprime le HTML et normalise les espaces."""
    if text is None:
        return ""

    cleaned_text = html.unescape(str(text))
    cleaned_text = re.sub(r"<[^>]+>", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def html_to_markdown_text(text: str | None) -> str:
    """Convertit le HTML utile en Markdown simple."""
    if text is None:
        return ""

    markdown_text = html.unescape(str(text)).strip()
    if not markdown_text:
        return ""

    markdown_text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    markdown_text = re.sub(
        r'<a\b[^>]*>\s*([^<]*?)\s*:\s*</a>\s*<a\b[^>]*>\s*https?://[^<]+\s*</a>',
        lambda match: normalize_text(match.group(1)),
        markdown_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def replace_anchor(match: re.Match[str]) -> str:
        anchor_text = normalize_text(re.sub(r"<[^>]+>", "", match.group(1)))
        if re.fullmatch(r"https?://\S+", anchor_text):
            return ""
        return anchor_text

    markdown_text = re.sub(
        r"<a\b[^>]*>(.*?)</a>",
        replace_anchor,
        markdown_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    replacements = [
        (r"<br\s*/?>", "\n"),
        (r"</p\s*>", "\n\n"),
        (r"<p\b[^>]*>", ""),
        (r"<ul\b[^>]*>", "\n"),
        (r"</ul\s*>", "\n"),
        (r"<ol\b[^>]*>", "\n"),
        (r"</ol\s*>", "\n"),
        (r"<li\b[^>]*>", "- "),
        (r"</li\s*>", "\n"),
        (r"</?(strong|em)\b[^>]*>", ""),
    ]

    for pattern, replacement in replacements:
        markdown_text = re.sub(
            pattern,
            replacement,
            markdown_text,
            flags=re.IGNORECASE,
        )

    markdown_text = re.sub(r"<[^>]+>", "", markdown_text)

    cleaned_lines: list[str] = []
    for line in markdown_text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_line = re.sub(r"\s+([,.!?])", r"\1", cleaned_line)

        if cleaned_line.startswith("- "):
            cleaned_line = "- " + cleaned_line[2:].lstrip()

        cleaned_lines.append(cleaned_line)

    normalized_lines: list[str] = []
    for line in cleaned_lines:
        if not line:
            if not normalized_lines or normalized_lines[-1] == "":
                continue
            normalized_lines.append("")
            continue

        if (
            line.startswith("- ")
            and normalized_lines
            and normalized_lines[-1] == ""
            and len(normalized_lines) >= 2
            and normalized_lines[-2].startswith("- ")
        ):
            normalized_lines.pop()

        normalized_lines.append(line)

    markdown_text = "\n".join(normalized_lines)
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

    return markdown_text.strip()


def normalize_text(text: str | None) -> str:
    """Normalise un texte simple."""
    if text is None:
        return ""

    normalized_text = html.unescape(str(text))
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    return normalized_text.strip()


def normalize_optional_text(text: str | None, default: str) -> str:
    """Retourne un texte nettoyé ou une valeur par défaut."""
    cleaned_text = clean_html(text)
    return cleaned_text if cleaned_text else default


def normalize_iso_datetime(value: str | None) -> str | None:
    """Retourne une date ISO normalisée ou None."""
    parsed_value = parse_event_datetime(value)
    if parsed_value is None:
        return None
    return parsed_value.isoformat()


def extract_coordinates(event: dict) -> tuple[float | None, float | None]:
    """Extrait latitude et longitude depuis location_coordinates."""
    coordinates = event.get("location_coordinates")
    if not isinstance(coordinates, dict):
        return None, None

    latitude = coordinates.get("lat")
    longitude = coordinates.get("lon")

    try:
        latitude_value = float(latitude) if latitude is not None else None
        longitude_value = float(longitude) if longitude is not None else None
    except (TypeError, ValueError):
        return None, None

    return latitude_value, longitude_value


def format_keywords(value) -> str:
    """Normalise les mots-clés OpenAgenda."""
    if value is None:
        return ""

    if isinstance(value, list):
        keywords = [normalize_text(item) for item in value]
    else:
        keywords = [normalize_text(value)]

    useful_keywords = [
        keyword
        for keyword in keywords
        if keyword and not keyword.lower().startswith("challenge-id=")
    ]

    return ", ".join(useful_keywords)


def extract_image_url(event: dict) -> str | None:
    """Retourne la meilleure URL d'image disponible."""
    for field_name in ["image", "thumbnail", "originalimage"]:
        image_value = normalize_text(event.get(field_name))
        if image_value:
            return image_value
    return None


def build_clean_description(event: dict) -> str:
    """Construit une description nettoyée sans dupliquer les métadonnées."""
    short_description = html_to_markdown_text(event.get("description_fr"))
    long_description = html_to_markdown_text(event.get("longdescription_fr"))
    conditions = html_to_markdown_text(event.get("conditions_fr"))

    if long_description and short_description in long_description:
        description = long_description
    elif long_description and short_description:
        description = f"{short_description}\n\n{long_description}"
    elif long_description:
        description = long_description
    elif short_description:
        description = short_description
    else:
        description = "Description non disponible"

    if conditions:
        description = f"{description}\n\nConditions : {conditions}"

    return description


def extract_french_label_from_json(value) -> str | None:
    """Extrait le libellé français d'un champ JSON OpenAgenda."""
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_value, dict):
        return None

    label = parsed_value.get("label")
    if isinstance(label, dict):
        cleaned_label = normalize_text(label.get("fr"))
        return cleaned_label or None

    return None

def clean_event(raw_event: dict) -> dict | None:
    """Nettoie un événement brut et retourne un format homogène."""
    raw_id = raw_event.get("uid")
    if raw_id is None or str(raw_id).strip() == "":
        raw_id = raw_event.get("slug")

    if raw_id is None or str(raw_id).strip() == "":
        return None

    event_id = str(raw_id).strip()
    latitude, longitude = extract_coordinates(raw_event)
    canonical_url = normalize_text(raw_event.get("canonicalurl"))
    online_url = normalize_text(raw_event.get("onlineaccesslink"))

    cleaned_event = {
        "event_id": event_id,
        "title": normalize_optional_text(
            raw_event.get("title_fr"),
            "Titre non disponible",
        ),
        "description": build_clean_description(raw_event),
        "conditions": html_to_markdown_text(raw_event.get("conditions_fr")) or None,
        "city": normalize_optional_text(
            raw_event.get("location_city"),
            "Ville non disponible",
        ),
        "location_name": normalize_optional_text(
            raw_event.get("location_name"),
            "Lieu non disponible",
        ),
        "address": normalize_optional_text(
            raw_event.get("location_address"),
            "Adresse non disponible",
        ),
        "start_date": normalize_iso_datetime(raw_event.get("firstdate_begin")),
        "end_date": normalize_iso_datetime(raw_event.get("lastdate_end"))
        or normalize_iso_datetime(raw_event.get("firstdate_end"))
        or normalize_iso_datetime(raw_event.get("firstdate_begin")),
        "keywords": format_keywords(raw_event.get("keywords_fr")),
        "url": canonical_url or online_url or None,
        "image_url": extract_image_url(raw_event),
        "latitude": latitude,
        "longitude": longitude,
        "source": normalize_optional_text(
            raw_event.get("originagenda_title"),
            "OpenAgenda",
        ),
        "attendance_mode": extract_french_label_from_json(
            raw_event.get("attendancemode")
        ),
        "status": extract_french_label_from_json(raw_event.get("status")),
    }

    return cleaned_event


def clean_events(raw_events: list[dict]) -> list[dict]:
    """Nettoie et déduplique une liste d'événements bruts."""
    cleaned_events: list[dict] = []
    seen_event_ids: set[str] = set()

    for raw_event in raw_events:
        cleaned_event = clean_event(raw_event)
        if cleaned_event is None:
            continue

        event_id = cleaned_event["event_id"]
        if event_id in seen_event_ids:
            continue

        seen_event_ids.add(event_id)
        cleaned_events.append(cleaned_event)

    return cleaned_events


def get_cleaning_stats(raw_events: list[dict], clean_events: list[dict]) -> dict:
    """Calcule quelques statistiques simples sur le nettoyage."""
    total_raw_events = len(raw_events)
    total_clean_events = len(clean_events)
    removed_events = total_raw_events - total_clean_events

    missing_description_count = sum(
        1 for event in clean_events if event["description"] == "Description non disponible"
    )
    missing_location_count = sum(
        1 for event in clean_events if event["location_name"] == "Lieu non disponible"
    )
    missing_coordinates_count = sum(
        1
        for event in clean_events
        if event["latitude"] is None or event["longitude"] is None
    )
    missing_url_count = sum(1 for event in clean_events if event["url"] is None)
    missing_image_count = sum(1 for event in clean_events if event["image_url"] is None)

    return {
        "total_raw_events": total_raw_events,
        "total_clean_events": total_clean_events,
        "removed_events": removed_events,
        "duplicate_or_invalid_events": removed_events,
        "missing_description_count": missing_description_count,
        "missing_location_count": missing_location_count,
        "missing_coordinates_count": missing_coordinates_count,
        "missing_url_count": missing_url_count,
        "missing_image_count": missing_image_count,
    }


def save_clean_events(events: list[dict], output_path: Path) -> Path:
    """Sauvegarde la liste nettoyée d'événements au format JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    return output_path
