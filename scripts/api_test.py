"""Test fonctionnel de l'API FastAPI d'Écho.

Ce script appelle une API déjà lancée localement (par défaut sur
``http://127.0.0.1:8000``) et vérifie le comportement des endpoints
principaux :

- ``GET /health`` → 200 et payload minimal attendu ;
- ``GET /metadata`` → 200 et présence des champs techniques ;
- ``POST /ask`` → 200 et structure ``question`` / ``answer`` /
  ``sources`` ;
- ``POST /rebuild`` sans ``confirm=true`` → 400 (refus attendu).

Ce script est un **test fonctionnel manuel** : il ne fait PAS partie de
la suite ``pytest`` (aucune fonction ``test_*`` n'est définie). Il
n'effectue pas d'appel Mistral coûteux par défaut : le rebuild réel
(``POST /rebuild`` avec ``confirm=true``) n'est lancé que si l'option
``--with-rebuild`` est passée.

Lancer l'API avant le script :

    poetry run uvicorn echo_app.api.main:app --reload

Puis, dans un autre terminal :

    poetry run python scripts/api_test.py
    poetry run python scripts/api_test.py --with-rebuild
"""

from __future__ import annotations

import argparse
import sys

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUESTION = "Quels événements autour de l'astronomie sont proposés ?"
ASK_TIMEOUT_SECONDS = 60
REBUILD_TIMEOUT_SECONDS = 600

METADATA_EXPECTED_KEYS = {
    "service",
    "rag_service_ready",
    "vector_store_available",
    "chunks_count",
    "top_k_default",
    "last_rebuild_at",
    "embedding_model",
    "generation_model",
}


def _run_check(label: str, expected_status: int, request_call) -> dict:
    """Exécute un appel HTTP et capture le résultat sous forme structurée."""
    response = request_call()
    try:
        body = response.json()
    except ValueError:
        body = None
    return {
        "label": label,
        "status_code": response.status_code,
        "expected_status": expected_status,
        "ok": response.status_code == expected_status,
        "body": body,
    }


def check_health(base_url: str) -> dict:
    """Vérifie ``GET /health`` (200 attendu, payload minimal)."""
    result = _run_check(
        "GET /health",
        expected_status=200,
        request_call=lambda: requests.get(f"{base_url}/health", timeout=5),
    )
    body = result["body"] or {}
    expected_keys = {"status", "service", "rag_service_ready"}
    result["ok"] = result["ok"] and expected_keys.issubset(body.keys())
    return result


def check_metadata(base_url: str) -> dict:
    """Vérifie ``GET /metadata`` (200 attendu, présence des 8 champs)."""
    result = _run_check(
        "GET /metadata",
        expected_status=200,
        request_call=lambda: requests.get(f"{base_url}/metadata", timeout=5),
    )
    body = result["body"] or {}
    has_all_keys = METADATA_EXPECTED_KEYS.issubset(body.keys())
    result["ok"] = result["ok"] and has_all_keys
    result["has_all_keys"] = has_all_keys
    return result


def check_ask(base_url: str, question: str) -> dict:
    """Vérifie ``POST /ask`` (200 + structure attendue)."""
    result = _run_check(
        "POST /ask",
        expected_status=200,
        request_call=lambda: requests.post(
            f"{base_url}/ask",
            json={"question": question},
            timeout=ASK_TIMEOUT_SECONDS,
        ),
    )
    body = result["body"] or {}
    expected_keys = {"question", "answer", "sources"}
    result["ok"] = result["ok"] and expected_keys.issubset(body.keys())
    return result


def check_rebuild_refused(base_url: str) -> dict:
    """Vérifie ``POST /rebuild`` sans ``confirm`` (400 attendu)."""
    return _run_check(
        "POST /rebuild (confirm=false)",
        expected_status=400,
        request_call=lambda: requests.post(
            f"{base_url}/rebuild",
            json={"confirm": False},
            timeout=5,
        ),
    )


def check_rebuild_confirmed(base_url: str) -> dict:
    """Vérifie ``POST /rebuild`` avec ``confirm=true`` (200 attendu).

    Cet appel déclenche un vrai rebuild côté serveur (coûteux). Il n'est
    exécuté que sur demande explicite (``--with-rebuild``).
    """
    return _run_check(
        "POST /rebuild (confirm=true)",
        expected_status=200,
        request_call=lambda: requests.post(
            f"{base_url}/rebuild",
            json={"confirm": True},
            timeout=REBUILD_TIMEOUT_SECONDS,
        ),
    )


def _print_result(result: dict) -> None:
    """Affiche une ligne lisible pour un résultat de check."""
    status_label = "OK  " if result["ok"] else "FAIL"
    print(
        f"[{status_label}] {result['label']} "
        f"→ HTTP {result['status_code']} (attendu {result['expected_status']})"
    )

    body = result["body"] or {}
    if result["label"] == "GET /health" and result["ok"]:
        print(f"        rag_service_ready : {body.get('rag_service_ready')}")
    elif result["label"] == "GET /metadata" and result["ok"]:
        print(
            f"        vector_store_available : {body.get('vector_store_available')}, "
            f"chunks_count : {body.get('chunks_count')}"
        )
        print(
            f"        embedding_model : {body.get('embedding_model')}, "
            f"generation_model : {body.get('generation_model')}"
        )
    elif result["label"] == "POST /ask" and result["ok"]:
        answer = (body.get("answer") or "").strip().replace("\n", " ")
        preview = answer[:120] + ("..." if len(answer) > 120 else "")
        sources_count = len(body.get("sources") or [])
        print(f"        answer (extrait) : {preview}")
        print(f"        sources retournées : {sources_count}")
    elif result["label"].startswith("POST /rebuild (confirm=true)") and result["ok"]:
        print(f"        chunks_count : {body.get('chunks_count')}")
        print(f"        last_rebuild_at : {body.get('last_rebuild_at')}")
    elif "rebuild" in result["label"] and not result["ok"]:
        print(f"        detail : {body.get('detail')}")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : exécute les checks et retourne 0 si tout est OK."""
    parser = argparse.ArgumentParser(
        description=(
            "Test fonctionnel de l'API FastAPI d'Écho. "
            "L'API doit déjà être lancée."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"URL de base de l'API (défaut : {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question à envoyer à POST /ask.",
    )
    parser.add_argument(
        "--with-rebuild",
        action="store_true",
        help=(
            "Lance aussi POST /rebuild avec confirm=true (déclenche un vrai "
            "rebuild côté serveur, coûteux)."
        ),
    )
    args = parser.parse_args(argv)

    results: list[dict] = []
    try:
        results.append(check_health(args.base_url))
        results.append(check_metadata(args.base_url))
        results.append(check_ask(args.base_url, args.question))
        results.append(check_rebuild_refused(args.base_url))
        if args.with_rebuild:
            results.append(check_rebuild_confirmed(args.base_url))
    except requests.RequestException as exc:
        print(f"Erreur de connexion à l'API ({args.base_url}) : {exc}")
        print(
            "L'API doit être lancée avant : "
            "poetry run uvicorn echo_app.api.main:app --reload"
        )
        return 2

    print(f"Cible : {args.base_url}\n")
    for result in results:
        _print_result(result)

    all_ok = all(r["ok"] for r in results)
    print()
    if all_ok:
        print("Résultat global : OK — l'API répond conformément à l'attendu.")
        return 0
    print("Résultat global : FAIL — au moins un endpoint a un comportement inattendu.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
