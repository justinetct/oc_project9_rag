"""Commande CLI de reconstruction du vector store FAISS LangChain.

La logique de reconstruction est portée par
``echo_app.indexing.rebuild.rebuild_index`` et peut être appelée
directement depuis l'API FastAPI. Ce script reste le point d'entrée
ligne de commande et se contente d'appeler la fonction puis d'afficher
les statistiques retournées.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer echo_app.* et src.* quand le script est exécuté depuis scripts/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.indexing.rebuild import (  # noqa: E402
    load_documents,
    rebuild_index,
)

__all__ = ["load_documents", "rebuild_index"]


def main() -> None:
    """Reconstruit l'index et affiche les statistiques principales."""
    try:
        stats = rebuild_index()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Nombre de documents chargés : {stats['documents_count']}")
    print(f"Nombre de chunks construits : {stats['chunks_count']}")
    print(f"Nombre d'embeddings générés : {stats['embeddings_count']}")
    print(f"Dimension de l'index : {stats['embedding_dimension']}")
    print(f"Nombre de vecteurs dans l'index : {stats['vectors_count']}")
    print(f"Index sauvegardé : {stats['index_file']}")
    print(f"Docstore sauvegardé : {stats['docstore_file']}")
    print(f"Reconstruction terminée à : {stats['last_rebuild_at']}")


if __name__ == "__main__":
    main()
