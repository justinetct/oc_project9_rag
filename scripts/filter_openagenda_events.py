"""Script simple de filtrage temporel des événements OpenAgenda."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Permet d'importer src.* quand le script est exécuté depuis le dossier scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    OPENAGENDA_REFERENCE_DATE,
    PATHS,
    PROCESSED_EVENTS_FILTERED_FILENAME,
    RAW_OPENAGENDA_EVENTS_FILENAME,
)
from src.preprocessing import (
    filter_events_by_date,
    get_date_filtering_stats,
    save_filtered_events,
)


def main() -> None:
    """Filtre les événements bruts avec la date de référence du POC."""
    input_path = PATHS.data_raw / RAW_OPENAGENDA_EVENTS_FILENAME
    if not input_path.exists():
        print(
            "Fichier brut introuvable. Lancez d'abord : "
            "poetry run python scripts/fetch_openagenda_events.py"
        )
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    reference_datetime = datetime.fromisoformat(OPENAGENDA_REFERENCE_DATE).replace(
        tzinfo=timezone.utc
    )
    filtered_events = filter_events_by_date(events, reference_datetime)
    stats = get_date_filtering_stats(events, filtered_events)

    output_path = PATHS.data_processed / PROCESSED_EVENTS_FILTERED_FILENAME
    save_filtered_events(filtered_events, output_path)

    print(f"Nombre d'événements avant filtrage : {stats['total_events']}")
    print(f"Nombre d'événements après filtrage : {stats['kept_events']}")
    print(f"Nombre d'événements exclus : {stats['excluded_events']}")
    print(f"Fichier sauvegardé : data/processed/{output_path.name}")


if __name__ == "__main__":
    main()
