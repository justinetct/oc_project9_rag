"""Tests simples pour le script principal de reconstruction de l'index."""

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rebuild_index import load_documents, rebuild_index  # noqa: E402


def test_load_documents_reads_jsonl_lines(tmp_path) -> None:
    """load_documents doit lire chaque ligne JSON du fichier JSONL."""
    documents_path = tmp_path / "events_documents.jsonl"
    documents_path.write_text(
        "\n".join(
            [
                json.dumps({"event_id": "1", "document_text": "Doc A"}),
                "",
                json.dumps({"event_id": "2", "document_text": "Doc B"}),
            ]
        ),
        encoding="utf-8",
    )

    documents = load_documents(documents_path)

    assert documents == [
        {"event_id": "1", "document_text": "Doc A"},
        {"event_id": "2", "document_text": "Doc B"},
    ]


def test_load_documents_raises_when_file_missing(tmp_path) -> None:
    """Un fichier absent doit lever FileNotFoundError."""
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_path)


def test_rebuild_index_is_callable() -> None:
    """rebuild_index doit être exposé comme fonction utilisable."""
    assert callable(rebuild_index)
