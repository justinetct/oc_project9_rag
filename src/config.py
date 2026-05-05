"""Configuration commune du projet."""

import os

from dotenv import load_dotenv


load_dotenv()


def get_required_env(name: str) -> str:
    """Retourne une variable d'environnement obligatoire.

    Une erreur explicite est levée si la variable est absente ou vide.
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(f"La variable d'environnement {name} est requise.")
    return value
