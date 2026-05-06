"""Configuration centrale du projet OC-P9-RAG.

Ce module regroupe les constantes partagées par le projet : chemins locaux,
paramètres de collecte OpenAgenda et champs utiles pour la future indexation RAG.
Il ne contient pas de logique métier complexe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_NAME = "OC-P9-RAG"
SEED = 42

OPENAGENDA_DATASET_ID = "evenements-publics-openagenda"
OPENAGENDA_BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    f"{OPENAGENDA_DATASET_ID}/records"
)

OPENAGENDA_CITIES = [
    # Bassin d'Arcachon
    "Arcachon",
    "La Teste-de-Buch",
    "Pyla-sur-Mer",
    "Gujan-Mestras",
    "Le Teich",
    "Biganos",
    "Audenge",
    "Lanton",
    "Andernos-les-Bains",
    "Arès",
    "Lège-Cap-Ferret",

    # Val de l'Eyre et communes proches
    "Mios",
    "Marcheprime",
    "Salles",
    "Belin-Béliet",
    "Le Barp",
    "Lugos",
    "Saint-Magne",
]

OPENAGENDA_DEPARTMENT = "Gironde"
OPENAGENDA_REGION = "Nouvelle-Aquitaine"
OPENAGENDA_COUNTRY_CODES = ["FR", "fr"]
OPENAGENDA_REFERENCE_DATE = "2026-05-01"
OPENAGENDA_DATE_FILTER = f"lastdate_end >= date'{OPENAGENDA_REFERENCE_DATE}'"
OPENAGENDA_ORDER_BY = "firstdate_begin asc"
OPENAGENDA_PAGE_SIZE = 100

OPENAGENDA_USEFUL_FIELDS = [
    "uid",
    "slug",
    "canonicalurl",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "daterange_fr",
    "firstdate_begin",
    "firstdate_end",
    "lastdate_begin",
    "lastdate_end",
    "timings",
    "location_name",
    "location_address",
    "location_postalcode",
    "location_city",
    "location_department",
    "location_region",
    "location_countrycode",
    "location_coordinates",
    "accessibility_label_fr",
    "age_min",
    "age_max",
    "registration",
    "onlineaccesslink",
    "originagenda_title",
]

METADATA_FIELDS = [
    "uid",
    "slug",
    "canonicalurl",
    "location_department",
    "location_region",
    "location_countrycode",
]

RAG_USEFUL_FIELDS = [
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "daterange_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_name",
    "location_address",
    "location_postalcode",
    "location_city",
    "location_coordinates",
    "accessibility_label_fr",
    "age_min",
    "age_max",
    "registration",
    "onlineaccesslink",
]

RAW_OPENAGENDA_SAMPLE_FILENAME = "sample_openagenda.json"
RAW_OPENAGENDA_EVENTS_FILENAME = "openagenda_events_raw.json"
PROCESSED_EVENTS_FILTERED_FILENAME = "events_filtered.json"
PROCESSED_OPENAGENDA_FILENAME = "openagenda_events_processed.csv"


@dataclass(frozen=True)
class Paths:
    """Chemins principaux du projet."""

    root: Path
    data: Path
    data_raw: Path
    data_processed: Path
    docs: Path
    notebooks: Path
    scripts: Path
    src: Path
    tests: Path
    faiss_index: Path


def get_paths() -> Paths:
    """Construit les chemins du projet à partir de l'emplacement de ce fichier."""
    here = Path(__file__).resolve()
    root = here.parents[1]
    data_dir = root / "data"

    return Paths(
        root=root,
        data=data_dir,
        data_raw=data_dir / "raw",
        data_processed=data_dir / "processed",
        docs=root / "docs",
        notebooks=root / "notebooks",
        scripts=root / "scripts",
        src=root / "src",
        tests=root / "tests",
        faiss_index=root / "faiss_index",
    )


PATHS = get_paths()


def get_required_env(name: str) -> str:
    """Retourne une variable d'environnement obligatoire.

    Une erreur explicite est levée si la variable est absente ou vide.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"La variable d'environnement {name} est requise.")
    return value
