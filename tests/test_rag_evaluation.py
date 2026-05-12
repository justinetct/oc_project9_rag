"""Tests unitaires pour les fonctions de calcul de l'évaluation RAG.

Ces tests vérifient uniquement les fonctions pures de scripts/08_evaluate_rag.py
(parsing, scoring, agrégation). Aucun appel réseau n'est effectué :
RagService.ask() n'est jamais invoqué ici.
"""

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.config import PATHS  # noqa: E402

# Le nom du script commence par un chiffre, donc on le charge via importlib
# plutôt qu'avec une instruction "import" classique.
_SCRIPT_PATH = PATHS.scripts / "08_evaluate_rag.py"
_spec = importlib.util.spec_from_file_location("evaluate_rag", _SCRIPT_PATH)
evaluate_rag = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_rag"] = evaluate_rag
_spec.loader.exec_module(evaluate_rag)


def test_split_semicolon_values_strips_and_filters() -> None:
    """Les valeurs vides ou blanches doivent être ignorées, les autres trimées."""
    assert evaluate_rag.split_semicolon_values("vélo; famille ;") == ["vélo", "famille"]
    assert evaluate_rag.split_semicolon_values(" ; ; ") == []
    assert evaluate_rag.split_semicolon_values(None) == []
    assert evaluate_rag.split_semicolon_values("") == []


def test_count_keyword_matches_is_case_insensitive() -> None:
    """La détection des mots-clés doit être insensible à la casse."""
    answer = "Une initiation à l'astronomie à Lanton, observation du ciel."
    expected = ["Astronomie", "lanton", "concert"]
    matched = evaluate_rag.count_keyword_matches(answer, expected)

    assert "Astronomie" in matched
    assert "lanton" in matched
    assert "concert" not in matched


def test_compute_keyword_match_rate_returns_half_when_one_of_two_found() -> None:
    """Si 1 mot-clé sur 2 est retrouvé, le taux doit être 0.5."""
    rate = evaluate_rag.compute_keyword_match_rate(["a"], ["a", "b"])
    assert rate == 0.5


def test_compute_keyword_match_rate_returns_zero_when_no_expected_keywords() -> None:
    """Sans mots-clés attendus, le taux est par convention 0.0."""
    assert evaluate_rag.compute_keyword_match_rate([], []) == 0.0


def test_extract_source_event_ids_reads_from_response() -> None:
    """extract_source_event_ids doit lire les event_id des sources."""
    response = {
        "sources": [
            {"event_id": "evt-1", "title": "A"},
            {"event_id": "evt-2", "title": "B"},
            {"event_id": "", "title": "ignored"},
            {"title": "no_event_id"},
        ]
    }
    event_ids = evaluate_rag.extract_source_event_ids(response)
    assert event_ids == ["evt-1", "evt-2"]


def test_extract_source_event_ids_handles_empty_response() -> None:
    """Une réponse sans sources doit retourner une liste vide."""
    assert evaluate_rag.extract_source_event_ids({}) == []
    assert evaluate_rag.extract_source_event_ids({"sources": []}) == []


def test_compute_event_recall_full_match_returns_one() -> None:
    """Si tous les event_ids attendus sont retrouvés, le rappel vaut 1.0."""
    recall = evaluate_rag.compute_event_recall(["evt-1", "evt-2"], ["evt-1", "evt-2"])
    assert recall == 1.0


def test_compute_event_recall_partial_match() -> None:
    """Si la moitié des event_ids sont retrouvés, le rappel vaut 0.5."""
    recall = evaluate_rag.compute_event_recall(["evt-1"], ["evt-1", "evt-2"])
    assert recall == 0.5


def test_compute_event_recall_no_expected_returns_one_when_no_sources() -> None:
    """Cas hors sujet : 1.0 si aucune source retournée, 0.0 sinon."""
    assert evaluate_rag.compute_event_recall([], []) == 1.0
    assert evaluate_rag.compute_event_recall(["evt-1"], []) == 0.0


def test_determine_status_out_of_scope_ok_when_no_sources() -> None:
    """Cas hors sujet sans source : status ok."""
    status = evaluate_rag.determine_status(
        keyword_match_rate=0.0,
        event_recall=1.0,
        expected_event_ids=[],
        sources_count=0,
        generated_answer="Je n'ai trouvé aucun événement.",
    )
    assert status == "ok"


def test_determine_status_out_of_scope_ok_when_refusal_message() -> None:
    """Cas hors sujet avec sources mais réponse de refus : status ok."""
    status = evaluate_rag.determine_status(
        keyword_match_rate=0.0,
        event_recall=0.0,
        expected_event_ids=[],
        sources_count=3,
        generated_answer="Je ne peux pas répondre, ne permet pas de répondre à cela.",
    )
    assert status == "ok"


def test_determine_status_out_of_scope_partial_when_no_refusal() -> None:
    """Cas hors sujet avec sources et pas de refus clair : status partial."""
    status = evaluate_rag.determine_status(
        keyword_match_rate=0.0,
        event_recall=0.0,
        expected_event_ids=[],
        sources_count=2,
        generated_answer="Voici quelques événements pour vous.",
    )
    assert status == "partial"


def test_determine_status_ok_with_event_ids() -> None:
    """Avec event_ids attendus : ok si keyword_rate >= 0.5 et recall > 0."""
    status = evaluate_rag.determine_status(
        keyword_match_rate=0.6,
        event_recall=0.5,
        expected_event_ids=["evt-1"],
        sources_count=1,
        generated_answer="Réponse contenant les mots-clés.",
    )
    assert status == "ok"


def test_determine_status_partial_with_event_ids() -> None:
    """Avec event_ids attendus : partial si l'une des métriques est positive."""
    status = evaluate_rag.determine_status(
        keyword_match_rate=0.2,
        event_recall=0.0,
        expected_event_ids=["evt-1"],
        sources_count=1,
        generated_answer="",
    )
    assert status == "partial"


def test_determine_status_ko_with_event_ids() -> None:
    """Avec event_ids attendus : ko si rien n'est retrouvé."""
    status = evaluate_rag.determine_status(
        keyword_match_rate=0.0,
        event_recall=0.0,
        expected_event_ids=["evt-1"],
        sources_count=1,
        generated_answer="Aucun match.",
    )
    assert status == "ko"


def test_summarize_results_counts_and_averages() -> None:
    """summarize_results doit agréger compteurs et moyennes."""
    results = [
        {"status": "ok", "keyword_match_rate": 1.0, "event_recall": 1.0},
        {"status": "ok", "keyword_match_rate": 0.6, "event_recall": 0.5},
        {"status": "partial", "keyword_match_rate": 0.2, "event_recall": 0.0},
        {"status": "ko", "keyword_match_rate": 0.0, "event_recall": 0.0},
    ]
    summary = evaluate_rag.summarize_results(results)

    assert summary["total"] == 4
    assert summary["ok"] == 2
    assert summary["partial"] == 1
    assert summary["ko"] == 1
    assert summary["average_keyword_match_rate"] == round((1.0 + 0.6 + 0.2 + 0.0) / 4, 3)
    assert summary["average_event_recall"] == round((1.0 + 0.5 + 0.0 + 0.0) / 4, 3)


def test_summarize_results_handles_empty_list() -> None:
    """Une liste vide doit retourner un résumé à zéro sans diviser par zéro."""
    summary = evaluate_rag.summarize_results([])
    assert summary == {
        "total": 0,
        "ok": 0,
        "partial": 0,
        "ko": 0,
        "average_keyword_match_rate": 0.0,
        "average_event_recall": 0.0,
    }
