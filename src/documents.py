"""Construction des documents textuels indexables pour le RAG."""

from __future__ import annotations

import json
from pathlib import Path

from src.preprocessing import normalize_text, parse_event_datetime


def format_date_for_text(value: str | None) -> str:
    """Retourne une date lisible pour le texte du document."""
    if value is None or not str(value).strip():
        return ""

    parsed_value = parse_event_datetime(value)
    if parsed_value is not None:
        return parsed_value.strftime("%Y-%m-%d %H:%M")

    return normalize_text(value)


def add_text_part(parts: list[str], label: str, value: str | None) -> None:
    """Ajoute une puce Markdown si la valeur est disponible."""
    if value is None:
        return

    cleaned_value = normalize_text(value)
    if cleaned_value:
        parts.append(f"- {label} : {cleaned_value}")


def build_event_document_text(event: dict) -> str:
    """Construit un texte Markdown lisible à partir d'un événement nettoyé."""
    title = normalize_text(event.get("title")) or "Titre non disponible"
    description = normalize_text(event.get("description"))

    lines = [f"# {title}"]

    if description:
        lines.extend(["", "## Description", description])

    info_parts: list[str] = []
    add_text_part(info_parts, "Lieu", event.get("location_name"))
    add_text_part(info_parts, "Ville", event.get("city"))
    add_text_part(info_parts, "Adresse", event.get("address"))

    start_date = format_date_for_text(event.get("start_date"))
    end_date = format_date_for_text(event.get("end_date"))
    if start_date and end_date:
        info_parts.append(f"- Dates : du {start_date} au {end_date}")
    elif start_date:
        info_parts.append(f"- Dates : à partir du {start_date}")
    elif end_date:
        info_parts.append(f"- Dates : jusqu'au {end_date}")

    add_text_part(info_parts, "Mots-clés", event.get("keywords"))

    conditions = normalize_text(event.get("conditions"))
    if conditions and "Conditions :" not in description:
        info_parts.append(f"- Conditions : {conditions}")

    add_text_part(info_parts, "Mode de participation", event.get("attendance_mode"))
    add_text_part(info_parts, "Statut", event.get("status"))

    if info_parts:
        lines.extend(["", "## Informations pratiques", *info_parts])

    source_parts: list[str] = []
    add_text_part(source_parts, "Organisateur", event.get("source"))

    if source_parts:
        lines.extend(["", "## Source", *source_parts])

    return "\n".join(lines)


def build_event_metadata(event: dict) -> dict:
    """Construit les métadonnées utiles du document."""
    metadata = {
        "event_id": event.get("event_id"),
        "title": event.get("title"),
        "city": event.get("city"),
        "location_name": event.get("location_name"),
        "start_date": event.get("start_date"),
        "end_date": event.get("end_date"),
        "url": event.get("url"),
        "image_url": event.get("image_url"),
        "latitude": event.get("latitude"),
        "longitude": event.get("longitude"),
        "source": event.get("source"),
    }

    if event.get("attendance_mode") not in (None, ""):
        metadata["attendance_mode"] = event.get("attendance_mode")
    if event.get("status") not in (None, ""):
        metadata["status"] = event.get("status")

    return metadata


def build_event_document(event: dict) -> dict | None:
    """Construit un document RAG à partir d'un événement nettoyé."""
    event_id = event.get("event_id")
    if event_id is None or not str(event_id).strip():
        return None

    return {
        "event_id": str(event_id).strip(),
        "document_text": build_event_document_text(event),
        "metadata": build_event_metadata(event),
    }


def build_event_documents(events: list[dict]) -> list[dict]:
    """Construit une liste de documents RAG."""
    documents: list[dict] = []

    for event in events:
        document = build_event_document(event)
        if document is not None:
            documents.append(document)

    return documents


def save_documents_jsonl(documents: list[dict], output_path: Path) -> Path:
    """Sauvegarde une liste de documents au format JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document, ensure_ascii=False) + "\n")

    return output_path


def get_document_stats(documents: list[dict]) -> dict:
    """Retourne quelques statistiques simples sur les documents générés."""
    if not documents:
        return {
            "total_documents": 0,
            "min_text_length": 0,
            "max_text_length": 0,
            "mean_text_length": 0.0,
            "empty_text_count": 0,
        }

    text_lengths = [len(document.get("document_text", "")) for document in documents]

    return {
        "total_documents": len(documents),
        "min_text_length": min(text_lengths),
        "max_text_length": max(text_lengths),
        "mean_text_length": round(sum(text_lengths) / len(text_lengths), 1),
        "empty_text_count": sum(1 for length in text_lengths if length == 0),
    }
