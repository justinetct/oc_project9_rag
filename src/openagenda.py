"""Fonctions OpenAgenda du projet.

Ce module regroupe les fonctions de construction du filtre métier, de
récupération paginée et de sauvegarde brute des événements OpenAgenda.
Les constantes utilisées par ces fonctions restent centralisées dans
`src.config`.
"""

import json
import time
from pathlib import Path
from typing import Any

import requests

from src.config import (
    OPENAGENDA_BASE_URL,
    OPENAGENDA_CITIES,
    OPENAGENDA_COUNTRY_CODES,
    OPENAGENDA_DATE_FILTER,
    OPENAGENDA_DEPARTMENT,
    OPENAGENDA_ORDER_BY,
    OPENAGENDA_PAGE_SIZE,
    OPENAGENDA_REGION,
)


def build_openagenda_where_clause() -> str:
    """Construit le filtre métier utilisé pour interroger OpenAgenda."""
    city_filter = ", ".join(f'"{city}"' for city in OPENAGENDA_CITIES)
    country_code_filter = ", ".join(f'"{code}"' for code in OPENAGENDA_COUNTRY_CODES)

    return "\n".join(
        [
            f'location_department = "{OPENAGENDA_DEPARTMENT}"',
            f'AND location_region = "{OPENAGENDA_REGION}"',
            f"AND location_countrycode IN ({country_code_filter})",
            f"AND location_city IN ({city_filter})",
            f"AND {OPENAGENDA_DATE_FILTER}",
        ]
    )


def fetch_openagenda_page(
    where_clause: str,
    limit: int = OPENAGENDA_PAGE_SIZE,
    offset: int = 0,
    order_by: str = OPENAGENDA_ORDER_BY,
) -> dict:
    """Récupère une page de résultats OpenAgenda."""
    params = {
        "limit": limit,
        "offset": offset,
        "where": where_clause,
        "order_by": order_by,
    }

    response = requests.get(OPENAGENDA_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_openagenda_events(
    where_clause: str,
    max_pages: int = 5,
    page_size: int = OPENAGENDA_PAGE_SIZE,
    pause_seconds: float = 0.2,
) -> list[dict]:
    """Récupère plusieurs pages d'événements OpenAgenda."""
    events: list[dict] = []

    for page_index in range(max_pages):
        offset = page_index * page_size
        page = fetch_openagenda_page(
            where_clause=where_clause,
            limit=page_size,
            offset=offset,
        )
        page_events = page.get("results", [])
        events.extend(page_events)

        if not page_events:
            break

        if len(page_events) < page_size:
            break

        if page_index < max_pages - 1:
            time.sleep(pause_seconds)

    return events


def _save_json(data: Any, output_path: Path) -> Path:
    """Sauvegarde des données JSON dans un fichier."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def save_openagenda_sample(data: dict, output_path: Path) -> Path:
    """Sauvegarde un échantillon brut de réponse OpenAgenda au format JSON."""
    return _save_json(data, output_path)


def save_openagenda_events(events: list[dict], output_path: Path) -> Path:
    """Sauvegarde une liste brute d'événements OpenAgenda au format JSON."""
    return _save_json(events, output_path)
