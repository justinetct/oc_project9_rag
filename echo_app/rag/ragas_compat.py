"""Compatibilité Ragas / mistralai 2.x.

Ragas tire ``instructor`` en dépendance transitive et celui-ci fait
``from mistralai import Mistral`` au moment du chargement
(``instructor/providers/mistral/client.py``). Or le projet utilise
``mistralai 2.4.4``, qui est un *namespace package* : la classe
``Mistral`` se trouve à ``mistralai.client.Mistral`` et n'est pas
exposée au top-level. Sans ce patch, tout ``import ragas`` échoue avec
``ImportError: cannot import name 'Mistral' from 'mistralai'``.

Ce helper expose ``Mistral`` à la racine de ``mistralai`` avant que
Ragas / instructor ne soient chargés. Il est uniquement utilisé par le
script d'évaluation ``scripts/08_evaluate_rag.py`` ; la chaîne RAG de
production ne dépend pas de Ragas et n'a pas besoin de ce patch.
"""

from __future__ import annotations


def patch_mistralai_namespace() -> None:
    """Expose ``Mistral`` au top-level du namespace ``mistralai``.

    Idempotent : si l'attribut existe déjà (mistralai 1.x ou patch déjà
    appliqué), la fonction ne fait rien.
    """
    import mistralai

    if hasattr(mistralai, "Mistral"):
        return

    from mistralai.client import Mistral

    mistralai.Mistral = Mistral
