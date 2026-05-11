"""Script manuel pour tester la recherche sémantique sur l'index FAISS local.

Ce script exécute quelques requêtes simples avec search_similar_events()
et affiche les résultats lisiblement. Il fait des appels API Mistral pour
générer l'embedding de chaque requête : l'index FAISS local doit donc avoir
été préalablement construit avec scripts/rebuild_index.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer echo_app.* quand le script est exécuté depuis scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.indexing.search import search_similar_events


QUERIES = [
    "astronomie",
    "concert à Andernos",
    "activité en famille",
    "exposition Bassin d'Arcachon",
    "événement gratuit",
]

EXCERPT_LENGTH = 200
SEPARATOR = "-" * 70


def format_date_range(metadata: dict) -> str:
    """Retourne une date lisible à partir des bornes start_date / end_date."""
    start = str(metadata.get("start_date") or "").strip()
    end = str(metadata.get("end_date") or "").strip()
    if start and end:
        return f"{start} → {end}"
    return start or end or "-"


def print_result(rank: int, result: dict) -> None:
    """Affiche un résultat de recherche lisiblement."""
    metadata = result.get("metadata") or {}
    title = metadata.get("title") or "(sans titre)"
    city = metadata.get("city") or "-"
    url = metadata.get("url") or "-"
    date_range = format_date_range(metadata)

    excerpt = (result.get("text") or "").replace("\n", " ").strip()
    if len(excerpt) > EXCERPT_LENGTH:
        excerpt = excerpt[:EXCERPT_LENGTH].rstrip() + "..."

    print(f"  {rank}. score={result['score']:.4f} (distance={result['distance']:.4f})")
    print(f"     Titre   : {title}")
    print(f"     Ville   : {city}")
    print(f"     Dates   : {date_range}")
    print(f"     URL     : {url}")
    print(f"     Extrait : {excerpt}")


def run_query(query: str, top_k: int = 5) -> None:
    """Exécute une requête et affiche les top_k résultats."""
    print(SEPARATOR)
    print(f"Requête : {query!r} (top_k={top_k})")
    print(SEPARATOR)

    results = search_similar_events(query, top_k=top_k)

    if not results:
        print("  Aucun résultat retourné.")
        return

    for rank, result in enumerate(results, start=1):
        print_result(rank, result)
        print()


def main() -> None:
    """Lance les requêtes de test prédéfinies."""
    try:
        for query in QUERIES:
            run_query(query, top_k=5)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Index FAISS introuvable. "
            "Exécutez d'abord poetry run python scripts/rebuild_index.py."
        ) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
