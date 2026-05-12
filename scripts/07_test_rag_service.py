"""Script manuel pour tester la chaîne RAG complète d'Écho.

Ce script instancie RagService et pose quelques questions pour vérifier que
la chaîne recherche FAISS + génération Mistral fonctionne de bout en bout.

Pré-requis :
- l'index FAISS local doit avoir été construit avec :
    poetry run python scripts/rebuild_index.py
- la clé MISTRAL_API_KEY doit être présente dans le fichier .env.

Ce script effectue un appel API Mistral par question (sauf pour les questions
qui ne retournent aucun chunk : la chaîne court-circuite alors le modèle).
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.rag import RagService


QUESTIONS = [
    "Quels événements autour de l'astronomie sont disponibles ?",
    "Je cherche une activité en famille",
    "Y a-t-il un concert à Andernos ?",
]

SEPARATOR = "=" * 70


def print_response(question: str, response: dict) -> None:
    """Affiche une réponse RAG lisiblement."""
    print(SEPARATOR)
    print(f"Question : {question}")
    print(SEPARATOR)
    print()
    print("Réponse :")
    print(response.get("answer") or "(réponse vide)")
    print()

    sources = response.get("sources") or []
    if not sources:
        print("Sources : (aucune)")
        return

    print("Sources :")
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "(sans titre)"
        city = source.get("city") or "-"
        start_date = source.get("start_date") or "-"
        url = source.get("url") or "-"
        print(f"  {index}. {title} — {city} ({start_date})")
        print(f"     URL : {url}")


def main() -> None:
    """Pose chaque question prédéfinie et affiche la réponse."""
    try:
        service = RagService(top_k=5)
        for question in QUESTIONS:
            response = service.ask(question)
            print_response(question, response)
            print()
    except FileNotFoundError as exc:
        raise SystemExit(
            "Index FAISS introuvable. "
            "Exécutez d'abord poetry run python scripts/rebuild_index.py."
        ) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
