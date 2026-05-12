"""Tests unitaires pour le jeu de test annoté du RAG d'Écho.

Ces tests vérifient la structure et la couverture minimale du fichier
``data/evaluation/qa_annotated.csv``. Aucun appel réseau n'est effectué.
"""

import sys
from pathlib import Path

import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))


QA_CSV_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "qa_annotated.csv"
)

EXPECTED_COLUMNS = {
    "question",
    "expected_answer",
    "expected_keywords",
    "expected_event_ids",
    "comment",
}


def _has_value(cell) -> bool:
    """Renvoie True si la cellule contient un texte non vide après strip."""
    return not pd.isna(cell) and str(cell).strip() != ""


def _load_dataset() -> pd.DataFrame:
    """Charge le CSV avec pandas en UTF-8."""
    return pd.read_csv(QA_CSV_PATH, encoding="utf-8")


def test_qa_csv_exists() -> None:
    """Le fichier annoté doit exister à l'emplacement attendu."""
    assert QA_CSV_PATH.exists(), f"Fichier introuvable : {QA_CSV_PATH}"


def test_qa_csv_has_expected_columns() -> None:
    """Les cinq colonnes attendues doivent être présentes dans le CSV."""
    df = _load_dataset()
    assert EXPECTED_COLUMNS.issubset(set(df.columns)), (
        f"Colonnes manquantes : {EXPECTED_COLUMNS - set(df.columns)}"
    )


def test_qa_csv_has_at_least_10_questions() -> None:
    """Le jeu doit contenir au moins 10 questions."""
    df = _load_dataset()
    assert len(df) >= 10


def test_qa_csv_has_exactly_15_rows() -> None:
    """Le jeu actuel contient 15 questions annotées."""
    df = _load_dataset()
    assert len(df) == 15


def test_qa_csv_question_and_answer_are_filled() -> None:
    """Chaque ligne doit avoir une question et une réponse attendue non vides."""
    df = _load_dataset()
    for index, row in df.iterrows():
        assert _has_value(row["question"]), f"question vide à la ligne {index}"
        assert _has_value(row["expected_answer"]), (
            f"expected_answer vide à la ligne {index}"
        )


def test_qa_csv_has_keywords_in_multiple_rows() -> None:
    """Plusieurs lignes doivent fournir des mots-clés attendus."""
    df = _load_dataset()
    rows_with_keywords = sum(_has_value(value) for value in df["expected_keywords"])
    assert rows_with_keywords >= 5


def test_qa_csv_has_at_least_one_event_id() -> None:
    """Au moins une ligne doit cibler explicitement un ou plusieurs event_ids."""
    df = _load_dataset()
    rows_with_event_ids = sum(_has_value(value) for value in df["expected_event_ids"])
    assert rows_with_event_ids >= 1


def test_qa_csv_has_edge_case_without_event_ids() -> None:
    """Au moins une ligne doit avoir expected_event_ids vide (cas limite / hors sujet)."""
    df = _load_dataset()
    rows_without_event_ids = sum(
        not _has_value(value) for value in df["expected_event_ids"]
    )
    assert rows_without_event_ids >= 1


def test_qa_csv_has_out_of_scope_question() -> None:
    """Au moins une ligne doit correspondre à un cas hors sujet."""
    df = _load_dataset()
    out_of_scope_terms = ("restaurant", "hors sujet")
    found = False
    for _, row in df.iterrows():
        blob = " ".join(
            str(row[col]) for col in ("question", "expected_keywords", "comment")
            if _has_value(row[col])
        ).lower()
        if any(term in blob for term in out_of_scope_terms):
            found = True
            break
    assert found, "Aucune ligne hors sujet (restaurant / hors sujet) détectée."
