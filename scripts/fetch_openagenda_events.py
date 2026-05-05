"""Script simple de collecte brute des événements OpenAgenda."""

import sys
from pathlib import Path

# Permet d'importer src.* quand le script est exécuté depuis le dossier scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, RAW_OPENAGENDA_EVENTS_FILENAME
from src.openagenda import (
    build_openagenda_where_clause,
    fetch_all_openagenda_events,
    save_openagenda_events,
)


def main() -> None:
    """Récupère puis sauvegarde les événements OpenAgenda bruts."""
    where_clause = build_openagenda_where_clause()
    events = fetch_all_openagenda_events(where_clause=where_clause, max_pages=10)

    output_path = PATHS.data_raw / RAW_OPENAGENDA_EVENTS_FILENAME
    save_openagenda_events(events, output_path)

    print(f"Nombre d'événements récupérés : {len(events)}")
    print(f"Fichier sauvegardé : data/raw/{output_path.name}")


if __name__ == "__main__":
    main()
