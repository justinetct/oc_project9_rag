"""Script simple de nettoyage des événements OpenAgenda filtrés."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permet d'importer src.* quand le script est exécuté depuis le dossier scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PATHS,
    PROCESSED_EVENTS_CLEAN_FILENAME,
    PROCESSED_EVENTS_FILTERED_FILENAME,
)
from src.preprocessing import clean_events, get_cleaning_stats, save_clean_events


def main() -> None:
    """Nettoie les événements filtrés et sauvegarde le résultat."""
    input_path = PATHS.data_processed / PROCESSED_EVENTS_FILTERED_FILENAME
    if not input_path.exists():
        print(
            "Fichier filtré introuvable. Lancez d'abord : "
            "poetry run python scripts/02_filter_openagenda_events.py"
        )
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as f:
        filtered_events = json.load(f)

    cleaned_events = clean_events(filtered_events)
    stats = get_cleaning_stats(filtered_events, cleaned_events)

    output_path = PATHS.data_processed / PROCESSED_EVENTS_CLEAN_FILENAME
    save_clean_events(cleaned_events, output_path)

    print(f"Nombre d'événements filtrés : {stats['total_raw_events']}")
    print(f"Nombre d'événements nettoyés : {stats['total_clean_events']}")
    print(f"Nombre d'événements supprimés : {stats['removed_events']}")
    print(
        "Nombre de descriptions manquantes : "
        f"{stats['missing_description_count']}"
    )
    print(f"Nombre de lieux manquants : {stats['missing_location_count']}")
    print(
        "Nombre de coordonnées manquantes : "
        f"{stats['missing_coordinates_count']}"
    )
    print(f"Nombre d'URL manquantes : {stats['missing_url_count']}")
    print(f"Nombre d'images manquantes : {stats['missing_image_count']}")
    print(f"Fichier sauvegardé : data/processed/{output_path.name}")


if __name__ == "__main__":
    main()
