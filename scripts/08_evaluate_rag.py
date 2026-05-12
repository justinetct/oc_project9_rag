"""Évaluation automatique simple du RAG d'Écho sur le jeu de test annoté.

Ce script :
- charge le jeu de questions annotées depuis data/evaluation/qa_annotated.csv ;
- exécute RagService.ask(question) pour chaque ligne (appel API Mistral réel) ;
- compare la réponse générée aux attentes via deux métriques simples :
    * keyword_match_rate : proportion de mots-clés attendus retrouvés dans
      la réponse générée (comparaison lowercase, sous-chaîne brute) ;
    * event_recall : proportion d'event_ids attendus retrouvés dans les
      sources retournées par le service ;
- attribue un status ok / partial / ko à chaque ligne via des règles
  simples, documentées dans determine_status() ;
- écrit les résultats par ligne dans data/evaluation/rag_evaluation_results.csv ;
- écrit un résumé agrégé dans data/evaluation/rag_evaluation_summary.json.

Pré-requis (exécution manuelle) :
- l'index FAISS doit avoir été reconstruit : poetry run python scripts/rebuild_index.py
- la clé MISTRAL_API_KEY doit être présente dans le fichier .env.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.rag import RagService  # noqa: E402
from src.config import PATHS  # noqa: E402


QA_DATASET_PATH = PATHS.data_evaluation / "qa_annotated.csv"
RESULTS_PATH = PATHS.data_evaluation / "rag_evaluation_results.csv"
SUMMARY_PATH = PATHS.data_evaluation / "rag_evaluation_summary.json"

RESULT_COLUMNS = [
    "question",
    "expected_answer",
    "generated_answer",
    "expected_keywords",
    "matched_keywords",
    "keyword_match_rate",
    "expected_event_ids",
    "matched_event_ids",
    "event_recall",
    "sources_count",
    "status",
    "comment",
]

# Phrases types renvoyées par le service quand il refuse de répondre. Elles
# permettent au status de considérer une absence d'invention comme "ok" même
# quand le LLM a quand même remonté quelques chunks.
REFUSAL_HINTS = (
    "aucun événement",
    "ne permet pas de répondre",
    "uniquement sur les événements",
    "je n'ai pas trouvé",
    "pas d'événement",
)


def split_semicolon_values(value) -> list[str]:
    """Découpe une chaîne séparée par ';' en valeurs non vides nettoyées."""
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def count_keyword_matches(answer: str, expected_keywords: list[str]) -> list[str]:
    """Retourne la liste des mots-clés attendus retrouvés dans la réponse."""
    answer_lower = (answer or "").lower()
    matched: list[str] = []
    for keyword in expected_keywords:
        if keyword and keyword.lower() in answer_lower:
            matched.append(keyword)
    return matched


def compute_keyword_match_rate(
    matched_keywords: list[str], expected_keywords: list[str]
) -> float:
    """Retourne le taux de mots-clés retrouvés, arrondi à 3 décimales."""
    if not expected_keywords:
        return 0.0
    return round(len(matched_keywords) / len(expected_keywords), 3)


def extract_source_event_ids(response: dict) -> list[str]:
    """Récupère les event_id présents dans les sources d'une réponse RAG."""
    sources = (response or {}).get("sources") or []
    event_ids: list[str] = []
    for source in sources:
        value = source.get("event_id") if isinstance(source, dict) else None
        if value is None:
            continue
        text = str(value).strip()
        if text:
            event_ids.append(text)
    return event_ids


def compute_event_recall(
    source_event_ids: list[str], expected_event_ids: list[str]
) -> float:
    """Calcule le rappel des event_ids attendus dans les sources.

    Règles :
    - si aucun event_id n'est attendu (cas hors sujet) : 1.0 si aucune
      source n'a été retournée, sinon 0.0 ;
    - sinon : nombre d'event_ids attendus retrouvés / nombre attendu.
    """
    if not expected_event_ids:
        return 1.0 if not source_event_ids else 0.0
    found = sum(1 for eid in expected_event_ids if eid in source_event_ids)
    return round(found / len(expected_event_ids), 3)


def _looks_like_refusal(answer: str) -> bool:
    """Vrai si la réponse contient une phrase de refus / hors contexte."""
    lowered = (answer or "").lower()
    return any(hint in lowered for hint in REFUSAL_HINTS)


def determine_status(
    keyword_match_rate: float,
    event_recall: float,
    expected_event_ids: list[str],
    sources_count: int,
    generated_answer: str = "",
) -> str:
    """Décide d'un status ok / partial / ko selon des règles simples.

    Cas hors sujet (expected_event_ids vide) :
    - "ok" si aucune source n'est retournée OU si la réponse indique
      clairement qu'elle ne peut pas répondre hors contexte ;
    - "partial" sinon.

    Cas avec event_ids attendus :
    - "ok" si keyword_match_rate >= 0.5 ET event_recall > 0 ;
    - "partial" si keyword_match_rate > 0 OU event_recall > 0 ;
    - "ko" sinon.
    """
    if not expected_event_ids:
        if sources_count == 0 or _looks_like_refusal(generated_answer):
            return "ok"
        return "partial"

    if keyword_match_rate >= 0.5 and event_recall > 0:
        return "ok"
    if keyword_match_rate > 0 or event_recall > 0:
        return "partial"
    return "ko"


def evaluate_row(row: dict, rag_service: RagService) -> dict:
    """Évalue une ligne du jeu de test annoté et retourne un dict de résultats."""
    question = (row.get("question") or "").strip()
    expected_keywords = split_semicolon_values(row.get("expected_keywords"))
    expected_event_ids = split_semicolon_values(row.get("expected_event_ids"))

    response = rag_service.ask(question)
    generated_answer = response.get("answer") or ""
    source_event_ids = extract_source_event_ids(response)
    sources_count = len(response.get("sources") or [])

    matched_keywords = count_keyword_matches(generated_answer, expected_keywords)
    keyword_match_rate = compute_keyword_match_rate(matched_keywords, expected_keywords)

    matched_event_ids = [eid for eid in expected_event_ids if eid in source_event_ids]
    event_recall = compute_event_recall(source_event_ids, expected_event_ids)

    status = determine_status(
        keyword_match_rate=keyword_match_rate,
        event_recall=event_recall,
        expected_event_ids=expected_event_ids,
        sources_count=sources_count,
        generated_answer=generated_answer,
    )

    return {
        "question": question,
        "expected_answer": row.get("expected_answer") or "",
        "generated_answer": generated_answer,
        "expected_keywords": ";".join(expected_keywords),
        "matched_keywords": ";".join(matched_keywords),
        "keyword_match_rate": keyword_match_rate,
        "expected_event_ids": ";".join(expected_event_ids),
        "matched_event_ids": ";".join(matched_event_ids),
        "event_recall": event_recall,
        "sources_count": sources_count,
        "status": status,
        "comment": row.get("comment") or "",
    }


def summarize_results(results: list[dict]) -> dict:
    """Agrège les résultats : compte par status et moyennes des métriques."""
    total = len(results)
    counts = {"ok": 0, "partial": 0, "ko": 0}
    for result in results:
        status = result.get("status")
        if status in counts:
            counts[status] += 1

    if total == 0:
        return {
            "total": 0,
            "ok": 0,
            "partial": 0,
            "ko": 0,
            "average_keyword_match_rate": 0.0,
            "average_event_recall": 0.0,
        }

    avg_kw = sum(r.get("keyword_match_rate", 0.0) for r in results) / total
    avg_ev = sum(r.get("event_recall", 0.0) for r in results) / total

    return {
        "total": total,
        "ok": counts["ok"],
        "partial": counts["partial"],
        "ko": counts["ko"],
        "average_keyword_match_rate": round(avg_kw, 3),
        "average_event_recall": round(avg_ev, 3),
    }


def _load_dataset() -> list[dict]:
    """Charge le CSV annoté en liste de dictionnaires."""
    if not QA_DATASET_PATH.exists():
        raise SystemExit(
            f"Fichier introuvable : {QA_DATASET_PATH}. "
            "Le jeu de test annoté doit être présent pour lancer l'évaluation."
        )

    with QA_DATASET_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _write_results_csv(results: list[dict]) -> None:
    """Écrit les résultats détaillés au format CSV (lisible à la main)."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _write_summary_json(summary: dict) -> None:
    """Écrit le résumé agrégé au format JSON."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Exécute l'évaluation complète et écrit les fichiers de résultats."""
    rows = _load_dataset()
    if not rows:
        raise SystemExit("Le jeu de test annoté ne contient aucune ligne.")

    rag_service = RagService(top_k=5)

    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        question_preview = (row.get("question") or "").strip()[:70]
        print(f"[{index}/{len(rows)}] {question_preview}")
        try:
            result = evaluate_row(row, rag_service)
        except FileNotFoundError as exc:
            raise SystemExit(
                "Index FAISS introuvable. "
                "Exécutez d'abord poetry run python scripts/rebuild_index.py."
            ) from exc
        results.append(result)
        print(
            f"    status={result['status']} | "
            f"keyword_match_rate={result['keyword_match_rate']} | "
            f"event_recall={result['event_recall']} | "
            f"sources={result['sources_count']}"
        )

    summary = summarize_results(results)

    _write_results_csv(results)
    _write_summary_json(summary)

    print("\n--- Résumé ---")
    print(f"  Total       : {summary['total']}")
    print(f"  OK          : {summary['ok']}")
    print(f"  Partial     : {summary['partial']}")
    print(f"  KO          : {summary['ko']}")
    print(f"  keyword_match_rate moyen : {summary['average_keyword_match_rate']}")
    print(f"  event_recall moyen       : {summary['average_event_recall']}")
    print(f"\nDétails : {RESULTS_PATH}")
    print(f"Résumé  : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
