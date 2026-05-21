"""Évaluation Ragas du RAG d'Écho sur le jeu de test annoté.

- format inspiré du cours OpenClassrooms : ``datasets.Dataset.from_dict``
  avec les colonnes ``question``, ``answer``, ``contexts``, ``ground_truth`` ;
- Ragas est l'évaluation principale (``faithfulness``, ``answer_relevancy``,
  ``context_precision``, ``context_recall``) ;
- les métriques maison (``keyword_match_rate``, ``event_recall``, ``status``…)
  sont conservées comme colonnes complémentaires ;
- une exécution complète : 15 appels RAG puis 60 jobs Ragas (15 × 4 métriques) ;
- la sortie console de la dernière exécution est versionnée dans
  ``data/evaluation/rag_evaluation.log``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path


# Réduit le bruit HF Hub avant les imports langchain-mistralai.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.rag import RagService  # noqa: E402
from echo_app.rag.ragas_compat import patch_mistralai_namespace  # noqa: E402
from src.config import PATHS  # noqa: E402


QA_DATASET_PATH = PATHS.data_evaluation / "qa_annotated.csv"
RESULTS_PATH = PATHS.data_evaluation / "rag_evaluation_results.csv"
SUMMARY_PATH = PATHS.data_evaluation / "rag_evaluation_summary.json"

# Modèles utilisés uniquement pour l'évaluation Ragas.
MISTRAL_JUDGE_MODEL = "mistral-large-latest"
MISTRAL_EMBED_MODEL = "mistral-embed"
# strictness=1 contourne un bug d'agrégation Ragas/langchain-mistralai 1.1.4
# sur n>1 complétions.
ANSWER_RELEVANCY_STRICTNESS = 1

# Exécution sérielle : plus lente, mais plus stable avec les quotas Mistral.
RAGAS_MAX_WORKERS = 1

RAGAS_METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

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
    *RAGAS_METRIC_COLUMNS,
    "status",
    "comment",
]

# Indices simples pour détecter une réponse de refus / hors contexte.
REFUSAL_HINTS = (
    "aucun événement",
    "ne permet pas de répondre",
    "uniquement sur les événements",
    "je n'ai pas trouvé",
    "pas d'événement",
)


# --- Métriques maison (fonctions pures, sans appel réseau) -----------------


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
) -> float | None:
    """Rappel des event_ids attendus dans les sources.

    None si aucun événement n'est attendu, car le rappel n'est pas applicable.
    """
    if not expected_event_ids:
        return None

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
    """Status ok / partial / ko selon les seuils sur keyword_match_rate et event_recall."""
    if not expected_event_ids:
        if sources_count == 0 or _looks_like_refusal(generated_answer):
            return "ok"
        return "partial"

    if keyword_match_rate >= 0.5 and event_recall > 0:
        return "ok"
    if keyword_match_rate > 0 or event_recall > 0:
        return "partial"
    return "ko"


def score_response(row: dict, response: dict) -> dict:
    """Calcule les métriques maison à partir d'une réponse RAG déjà obtenue."""
    question = (row.get("question") or "").strip()
    expected_keywords = split_semicolon_values(row.get("expected_keywords"))
    expected_event_ids = split_semicolon_values(row.get("expected_event_ids"))

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


def evaluate_row(row: dict, rag_service: RagService) -> dict:
    """Évalue une ligne via RagService.ask() + métriques maison (rétro-compat tests)."""
    response = rag_service.ask((row.get("question") or "").strip())
    return score_response(row, response)


def summarize_results(results: list[dict]) -> dict:
    """Agrège les résultats maison : compteurs par status + moyennes."""
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
            "average_event_recall": None,
        }

    keyword_values = [
        r.get("keyword_match_rate")
        for r in results
        if not _is_missing(r.get("keyword_match_rate"))
    ]
    event_values = [
        r.get("event_recall")
        for r in results
        if not _is_missing(r.get("event_recall"))
    ]

    return {
        "total": total,
        "ok": counts["ok"],
        "partial": counts["partial"],
        "ko": counts["ko"],
        "average_keyword_match_rate": round(sum(keyword_values) / len(keyword_values), 3)
        if keyword_values
        else None,
        "average_event_recall": round(sum(event_values) / len(event_values), 3)
        if event_values
        else None,
    }


def _is_missing(value) -> bool:
    """Vrai si la valeur Ragas est manquante (None ou NaN)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def merge_scores(custom_results: list[dict], ragas_df) -> list[dict]:
    """Joint les scores Ragas aux dicts maison (arrondi à 3 décimales, NaN → None)."""
    if len(custom_results) != len(ragas_df):
        raise ValueError(
            f"Désaccord d'ordre : {len(custom_results)} résultats maison "
            f"vs {len(ragas_df)} lignes Ragas."
        )

    merged: list[dict] = []
    for index, base in enumerate(custom_results):
        enriched = dict(base)
        row = ragas_df.iloc[index]
        for column in RAGAS_METRIC_COLUMNS:
            value = row.get(column) if column in ragas_df.columns else None
            enriched[column] = None if _is_missing(value) else round(float(value), 3)
        merged.append(enriched)
    return merged


def summarize_ragas_results(results: list[dict]) -> dict:
    """Résumé agrégé : compteurs status + moyennes maison et Ragas (None si tout manquant)."""
    total = len(results)
    counts = {"ok": 0, "partial": 0, "ko": 0}
    for result in results:
        status = result.get("status")
        if status in counts:
            counts[status] += 1

    def average(column: str):
        values = [r.get(column) for r in results if not _is_missing(r.get(column))]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    summary = {
        "total": total,
        "ok": counts["ok"],
        "partial": counts["partial"],
        "ko": counts["ko"],
        "average_keyword_match_rate": average("keyword_match_rate"),
        "average_event_recall": average("event_recall"),
    }
    for metric in RAGAS_METRIC_COLUMNS:
        summary[f"average_{metric}"] = average(metric)
    return summary


# --- Pipeline Ragas (appelle Mistral) ------------------------


def run_rag_inference(
    rows: list[dict], rag_service: RagService
) -> list[dict]:
    """Étape 1 — interroge le RAG sur chaque question et retourne les réponses brutes."""
    responses: list[dict] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        question = (row.get("question") or "").strip()
        expected_event_ids = split_semicolon_values(row.get("expected_event_ids"))
        cas = "hors-sujet" if not expected_event_ids else f"{len(expected_event_ids)} events"

        preview = question[:70]
        print(f"[{index}/{total}] {preview}")

        response = rag_service.ask(question, include_contexts=True)
        responses.append(response)

        chunks = len(response.get("contexts") or [])
        sources = len(response.get("sources") or [])
        scored = score_response(row, response)
        print(
            f"    chunks={chunks} | sources={sources} | "
            f"attendu={cas} | status={scored['status']}"
        )

    return responses


def build_ragas_dataset(rows: list[dict], responses: list[dict]):
    """Construit le ``Dataset`` Ragas : question / answer / contexts / ground_truth."""
    from datasets import Dataset

    data = {
        "question": [(row.get("question") or "").strip() for row in rows],
        "answer": [resp.get("answer") or "" for resp in responses],
        "contexts": [resp.get("contexts") or [] for resp in responses],
        "ground_truth": [(row.get("expected_answer") or "").strip() for row in rows],
    }
    return Dataset.from_dict(data)


def build_ragas_judge():
    """Construit le LLM juge Mistral + les embeddings (échoue si MISTRAL_API_KEY absente)."""
    if not os.environ.get("MISTRAL_API_KEY"):
        raise SystemExit(
            "MISTRAL_API_KEY introuvable dans l'environnement. "
            "Ajoutez la clé dans .env avant de lancer l'évaluation Ragas."
        )

    patch_mistralai_namespace()

    from langchain_mistralai.chat_models import ChatMistralAI
    from langchain_mistralai.embeddings import MistralAIEmbeddings

    api_key = os.environ["MISTRAL_API_KEY"]
    llm = ChatMistralAI(
        mistral_api_key=api_key,
        model=MISTRAL_JUDGE_MODEL,
        temperature=0,
    )
    embeddings = MistralAIEmbeddings(
        mistral_api_key=api_key,
        model=MISTRAL_EMBED_MODEL,
    )
    return llm, embeddings


def build_ragas_metrics() -> list:
    """Liste des 4 métriques Ragas utilisées."""
    patch_mistralai_namespace()

    # API legacy utilisée pour rester compatible avec ragas.evaluate().
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            AnswerRelevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    return [
        faithfulness,
        AnswerRelevancy(strictness=ANSWER_RELEVANCY_STRICTNESS),
        context_precision,
        context_recall,
    ]


def run_ragas_evaluation(dataset, llm, embeddings, metrics):
    """Étape 2 — lance ``ragas.evaluate`` et retourne le DataFrame des scores."""
    patch_mistralai_namespace()

    # nest_asyncio évite les conflits d'event loop en script/notebook.
    import nest_asyncio

    nest_asyncio.apply()

    from ragas import evaluate
    from ragas.run_config import RunConfig

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS),
        raise_exceptions=False,
    )
    return result.to_pandas()


# --- Fichiers d'entrée / sortie --------------------------------------------


def _load_dataset() -> list[dict]:
    """Charge le CSV annoté en liste de dictionnaires."""
    if not QA_DATASET_PATH.exists():
        raise SystemExit(
            f"Fichier introuvable : {QA_DATASET_PATH}. "
            "Le jeu de test annoté doit être présent pour lancer l'évaluation."
        )

    with QA_DATASET_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


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


# --- Orchestration ---------------------------------------------------------


def main() -> None:
    """Exécute l'évaluation Ragas complète et écrit les fichiers de résultats."""
    rows = _load_dataset()
    if not rows:
        raise SystemExit("Le jeu de test annoté ne contient aucune ligne.")

    rag_service = RagService(top_k=5)
    total_questions = len(rows)
    ragas_jobs = total_questions * len(RAGAS_METRIC_COLUMNS)

    print(
        "Étape 1/3 : interrogation du RAG sur chaque question "
        f"({total_questions} appels Mistral)..."
    )
    try:
        responses = run_rag_inference(rows, rag_service)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Index FAISS introuvable. "
            "Exécutez d'abord poetry run python scripts/rebuild_index.py."
        ) from exc

    # Scoring maison : pas d'appel LLM, juste du calcul local.
    custom_results = [
        score_response(row, response) for row, response in zip(rows, responses)
    ]

    print(
        f"\nÉtape 2/3 : évaluation Ragas — {total_questions} questions "
        f"× 4 métriques = {ragas_jobs} jobs (LLM juge Mistral, plusieurs "
        "appels par job possibles)..."
    )
    llm, embeddings = build_ragas_judge()
    metrics = build_ragas_metrics()
    dataset = build_ragas_dataset(rows, responses)
    ragas_df = run_ragas_evaluation(dataset, llm, embeddings, metrics)

    print("\nÉtape 3/3 : fusion des scores et écriture des résultats...")
    merged_results = merge_scores(custom_results, ragas_df)
    summary = summarize_ragas_results(merged_results)

    _write_results_csv(merged_results)
    _write_summary_json(summary)

    print("\n--- Résumé ---")
    print(f"  Total              : {summary['total']}")
    print(
        f"  OK / Partial / KO  : "
        f"{summary['ok']} / {summary['partial']} / {summary['ko']}"
    )
    print("  Métriques maison :")
    print(f"    keyword_match_rate moyen : {summary['average_keyword_match_rate']}")
    print(f"    event_recall moyen       : {summary['average_event_recall']}")
    print("  Métriques Ragas :")
    print(f"    faithfulness moyen       : {summary['average_faithfulness']}")
    print(f"    answer_relevancy moyen   : {summary['average_answer_relevancy']}")
    print(f"    context_precision moyen  : {summary['average_context_precision']}")
    print(f"    context_recall moyen     : {summary['average_context_recall']}")
    print(f"\nDétails : {RESULTS_PATH}")
    print(f"Résumé  : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
