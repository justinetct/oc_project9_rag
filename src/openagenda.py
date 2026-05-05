"""Fonctions utilitaires pour l’exploration de l’API OpenAgenda.

Ce module regroupe la logique spécifique à OpenAgenda : construction du filtre
de collecte et sauvegarde d’un échantillon brut. Les constantes utilisées par
ces fonctions restent centralisées dans `src.config`.
"""

import json
from pathlib import Path

from src.config import (
    OPENAGENDA_CITIES,
    OPENAGENDA_COUNTRY_CODES,
    OPENAGENDA_DATE_FILTER,
    OPENAGENDA_DEPARTMENT,
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


def save_openagenda_sample(data: dict, output_path: Path) -> Path:
    """Sauvegarde un échantillon brut de réponse OpenAgenda au format JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path
