"""Script simple de construction des documents textuels OpenAgenda."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permet d'importer src.* quand le script est exécuté depuis le dossier scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PATHS,
    PROCESSED_EVENTS_CLEAN_FILENAME,
    PROCESSED_EVENTS_DOCUMENTS_FILENAME,
)
from src.documents import (
    build_event_documents,
    get_document_stats,
    save_documents_jsonl,
)


def main() -> None:
    """Construit les documents textuels RAG à partir des événements nettoyés."""
    input_path = PATHS.data_processed / PROCESSED_EVENTS_CLEAN_FILENAME
    if not input_path.exists():
        print(
            "Fichier nettoyé introuvable. Lancez d'abord : "
            "poetry run python scripts/03_clean_openagenda_events.py"
        )
        raise SystemExit(1)

    with input_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    documents = build_event_documents(events)
    stats = get_document_stats(documents)

    output_path = PATHS.data_processed / PROCESSED_EVENTS_DOCUMENTS_FILENAME
    save_documents_jsonl(documents, output_path)

    print(f"Nombre d'événements nettoyés : {len(events)}")
    print(f"Nombre de documents créés : {stats['total_documents']}")
    print(f"Longueur minimale des textes : {stats['min_text_length']}")
    print(f"Longueur maximale des textes : {stats['max_text_length']}")
    print(f"Longueur moyenne des textes : {stats['mean_text_length']}")
    print(f"Documents vides : {stats['empty_text_count']}")
    print(f"Fichier sauvegardé : data/processed/{output_path.name}")


if __name__ == "__main__":
    main()
