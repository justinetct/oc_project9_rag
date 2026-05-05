"""Configuration de l'application Écho."""

import os

from src.config import get_required_env


MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

OPENAGENDA_BASE_URL = os.getenv(
    "OPENAGENDA_BASE_URL",
    "https://public.opendatasoft.com/api/explore/v2.1",
)
OPENAGENDA_DATASET_ID = os.getenv("OPENAGENDA_DATASET_ID")


def get_mistral_api_key() -> str:
    """Retourne la clé API Mistral lorsque l'appel au modèle est nécessaire."""
    return get_required_env("MISTRAL_API_KEY")
