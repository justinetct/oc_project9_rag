"""Tests unitaires pour les fonctions de calcul de l'évaluation RAG.

Ces tests vérifient uniquement les fonctions pures de scripts/08_evaluate_rag.py
(parsing, scoring, agrégation). Aucun appel réseau n'est effectué :
RagService.ask() n'est jamais invoqué ici.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


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


# --- Tests des nouvelles fonctions Ragas (sans appel réseau réel) ---


class _FakeRagService:
    """RagService factice : retourne une réponse pré-définie par question.

    Imite le contrat de ``RagService.ask(q, include_contexts=...)`` sans
    déclencher d'appel réseau. La séquence de réponses est fournie au
    constructeur dans l'ordre attendu par le test.
    """

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, bool]] = []

    def ask(self, question: str, *, include_contexts: bool = False) -> dict:
        self.calls.append((question, include_contexts))
        return self._responses.pop(0)


def test_score_response_combines_custom_metrics() -> None:
    """score_response doit combiner toutes les métriques maison et le status."""
    row = {
        "question": "Quels événements d'astronomie à Lanton ?",
        "expected_answer": "Astronomie à Lanton.",
        "expected_keywords": "astronomie;Lanton",
        "expected_event_ids": "evt-1",
        "comment": "thématique",
    }
    response = {
        "question": "Quels événements d'astronomie à Lanton ?",
        "answer": "Une initiation à l'astronomie à Lanton.",
        "sources": [{"event_id": "evt-1", "title": "Astro"}],
    }
    result = evaluate_rag.score_response(row, response)

    assert result["question"] == "Quels événements d'astronomie à Lanton ?"
    assert result["generated_answer"] == "Une initiation à l'astronomie à Lanton."
    assert result["expected_keywords"] == "astronomie;Lanton"
    assert result["matched_keywords"] == "astronomie;Lanton"
    assert result["keyword_match_rate"] == 1.0
    assert result["expected_event_ids"] == "evt-1"
    assert result["matched_event_ids"] == "evt-1"
    assert result["event_recall"] == 1.0
    assert result["sources_count"] == 1
    assert result["status"] == "ok"
    assert result["comment"] == "thématique"


def test_run_rag_inference_calls_ask_with_include_contexts(capsys) -> None:
    """run_rag_inference doit appeler ask(include_contexts=True) ligne par ligne."""
    rows = [
        {
            "question": "Question A ?",
            "expected_answer": "Réponse A attendue.",
            "expected_keywords": "alpha",
            "expected_event_ids": "evt-A",
            "comment": "",
        },
        {
            "question": "Question B ?",
            "expected_answer": "",
            "expected_keywords": "",
            "expected_event_ids": "",
            "comment": "hors sujet",
        },
    ]
    responses = [
        {
            "question": "Question A ?",
            "answer": "Réponse A générée.",
            "sources": [{"event_id": "evt-A", "title": "A"}],
            "contexts": ["chunk A1", "chunk A2"],
        },
        {
            "question": "Question B ?",
            "answer": "Je n'ai trouvé aucun événement.",
            "sources": [],
            "contexts": [],
        },
    ]
    fake_service = _FakeRagService(responses=list(responses))

    returned = evaluate_rag.run_rag_inference(rows, fake_service)

    assert fake_service.calls == [("Question A ?", True), ("Question B ?", True)]
    assert returned == responses

    # Le log doit être lisible : chunks / sources / type de cas / status.
    out = capsys.readouterr().out
    assert "[1/2] Question A ?" in out
    assert "chunks=2" in out
    assert "sources=1" in out
    assert "attendu=1 events" in out
    assert "[2/2] Question B ?" in out
    assert "attendu=hors-sujet" in out


def test_build_ragas_dataset_matches_course_columns() -> None:
    """Le dataset doit exposer les 4 colonnes Ragas attendues."""
    rows = [
        {
            "question": "Question A ?",
            "expected_answer": "Réponse A attendue.",
            "expected_keywords": "alpha",
            "expected_event_ids": "evt-A",
            "comment": "",
        },
    ]
    responses = [
        {
            "answer": "Réponse A générée.",
            "sources": [{"event_id": "evt-A"}],
            "contexts": ["chunk A1", "chunk A2"],
        }
    ]

    dataset = evaluate_rag.build_ragas_dataset(rows, responses)

    assert set(dataset.column_names) == {"question", "answer", "contexts", "ground_truth"}
    assert dataset[0] == {
        "question": "Question A ?",
        "answer": "Réponse A générée.",
        "contexts": ["chunk A1", "chunk A2"],
        "ground_truth": "Réponse A attendue.",
    }


def test_merge_scores_aligns_and_rounds_ragas_columns() -> None:
    """merge_scores doit joindre les scores Ragas ligne à ligne et arrondir."""
    import pandas as pd

    custom_results = [
        {"question": "Q1", "keyword_match_rate": 1.0, "event_recall": 1.0, "status": "ok"},
        {"question": "Q2", "keyword_match_rate": 0.5, "event_recall": 0.5, "status": "partial"},
    ]
    ragas_df = pd.DataFrame(
        [
            {
                "faithfulness": 0.9123,
                "answer_relevancy": 0.8567,
                "context_precision": 0.7,
                "context_recall": 1.0,
            },
            {
                "faithfulness": 0.4321,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": 0.5,
            },
        ]
    )

    merged = evaluate_rag.merge_scores(custom_results, ragas_df)

    assert merged[0]["faithfulness"] == 0.912
    assert merged[0]["answer_relevancy"] == 0.857
    assert merged[0]["context_precision"] == 0.7
    assert merged[0]["context_recall"] == 1.0
    # Les champs maison sont préservés
    assert merged[0]["question"] == "Q1"
    assert merged[0]["status"] == "ok"
    # Deuxième ligne
    assert merged[1]["faithfulness"] == 0.432
    assert merged[1]["answer_relevancy"] == 0.5


def test_merge_scores_serializes_nan_as_none() -> None:
    """Les NaN Ragas doivent être sérialisés en None pour produire un CSV vide."""
    import pandas as pd

    custom_results = [{"question": "Q1", "status": "partial"}]
    ragas_df = pd.DataFrame(
        [
            {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": float("nan"),
                "context_recall": float("nan"),
            }
        ]
    )

    merged = evaluate_rag.merge_scores(custom_results, ragas_df)

    assert merged[0]["faithfulness"] == 0.5
    assert merged[0]["answer_relevancy"] == 0.5
    assert merged[0]["context_precision"] is None
    assert merged[0]["context_recall"] is None


def test_merge_scores_raises_on_length_mismatch() -> None:
    """Un désaccord sur le nombre de lignes doit lever ValueError."""
    import pandas as pd

    with pytest.raises(ValueError, match="Désaccord d'ordre"):
        evaluate_rag.merge_scores(
            [{"question": "Q1"}],
            pd.DataFrame([{"faithfulness": 0.5}, {"faithfulness": 0.4}]),
        )


def test_summarize_ragas_results_counts_and_averages_both_families() -> None:
    """summarize_ragas_results agrège status + moyennes maison ET Ragas."""
    results = [
        {
            "status": "ok",
            "keyword_match_rate": 1.0,
            "event_recall": 1.0,
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 1.0,
        },
        {
            "status": "partial",
            "keyword_match_rate": 0.5,
            "event_recall": 0.5,
            "faithfulness": 0.5,
            "answer_relevancy": 0.6,
            "context_precision": 0.5,
            "context_recall": 0.5,
        },
    ]
    summary = evaluate_rag.summarize_ragas_results(results)

    assert summary["total"] == 2
    assert summary["ok"] == 1
    assert summary["partial"] == 1
    assert summary["ko"] == 0
    assert summary["average_keyword_match_rate"] == 0.75
    assert summary["average_event_recall"] == 0.75
    assert summary["average_faithfulness"] == 0.7
    assert summary["average_answer_relevancy"] == 0.7
    assert summary["average_context_precision"] == 0.6
    assert summary["average_context_recall"] == 0.75


def test_summarize_ragas_results_ignores_none_in_averages() -> None:
    """Les valeurs None doivent être exclues du calcul de moyenne."""
    results = [
        {
            "status": "ok",
            "keyword_match_rate": 1.0,
            "event_recall": 1.0,
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": None,  # cas hors sujet
        },
        {
            "status": "ok",
            "keyword_match_rate": 1.0,
            "event_recall": 1.0,
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
        },
    ]
    summary = evaluate_rag.summarize_ragas_results(results)

    # context_recall moyen = moyenne sur 1 valeur seulement
    assert summary["average_context_recall"] == 0.5
    assert summary["average_faithfulness"] == 0.75


def test_summarize_ragas_results_returns_none_when_all_missing() -> None:
    """Si toutes les valeurs d'une métrique sont absentes, la moyenne est None.

    Évite la lecture trompeuse "moyenne = 0.0 = très mauvais score" quand
    la métrique a en réalité systématiquement échoué (bug Ragas, rate
    limit Mistral, etc.).
    """
    results = [
        {
            "status": "ok",
            "keyword_match_rate": 1.0,
            "event_recall": 1.0,
            "faithfulness": 1.0,
            "answer_relevancy": None,
            "context_precision": 1.0,
            "context_recall": 1.0,
        },
        {
            "status": "ok",
            "keyword_match_rate": 0.5,
            "event_recall": 0.5,
            "faithfulness": 0.5,
            "answer_relevancy": None,
            "context_precision": 0.5,
            "context_recall": 0.5,
        },
    ]
    summary = evaluate_rag.summarize_ragas_results(results)

    assert summary["average_answer_relevancy"] is None
    assert summary["average_faithfulness"] == 0.75


def test_result_columns_match_spec() -> None:
    """Gel du contrat CSV : ordre exact des colonnes."""
    assert evaluate_rag.RESULT_COLUMNS == [
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
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "status",
        "comment",
    ]
