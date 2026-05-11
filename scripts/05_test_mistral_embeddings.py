"""Script manuel pour vérifier rapidement les embeddings Mistral."""

from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer echo_app.* quand le script est exécuté depuis le dossier scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.config import get_mistral_api_key
from echo_app.indexing.embeddings import embed_texts


def main() -> None:
    """Teste la génération d'embeddings Mistral sur quelques textes courts."""
    try:
        get_mistral_api_key()
    except ValueError as exc:
        raise SystemExit(
            "MISTRAL_API_KEY est absente. Configurez-la dans .env avant ce test."
        ) from exc

    sample_texts = [
        "Concert gratuit au parc Mauresque ce week-end.",
        "Exposition photo à Arcachon sur le patrimoine local.",
        "Atelier astronomie pour les familles à Lanton.",
    ]

    try:
        vectors = embed_texts(sample_texts)
    except RuntimeError as exc:
        raise SystemExit(
            "La génération des embeddings Mistral a échoué. "
            "Vérifiez la connectivité réseau et la validité de la configuration Mistral."
        ) from exc

    print(f"Nombre de textes vectorisés : {len(vectors)}")
    print(f"Dimension du premier vecteur : {len(vectors[0])}")
    print(f"Extrait du premier vecteur : {vectors[0][:5]}")


if __name__ == "__main__":
    main()
