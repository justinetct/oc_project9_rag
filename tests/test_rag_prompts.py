"""Tests unitaires pour les prompts métier du chatbot Écho.

Ces tests vérifient le contenu du prompt système et le format du prompt
utilisateur, sans aucun appel réseau.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.rag.prompts import RAG_SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def test_rag_system_prompt_contains_key_rules() -> None:
    """Le prompt système doit cadrer Écho avec les règles métier essentielles."""
    prompt_lower = RAG_SYSTEM_PROMPT.lower()

    assert "contexte" in prompt_lower
    assert "français" in prompt_lower
    assert "événements" in prompt_lower
    assert ("n'invente" in prompt_lower) or ("ne crée jamais" in prompt_lower)


def test_rag_system_prompt_mentions_persona_and_scope() -> None:
    """Le prompt doit identifier Écho et son périmètre culturel."""
    assert "Écho" in RAG_SYSTEM_PROMPT
    assert "Bassin d'Arcachon" in RAG_SYSTEM_PROMPT


def test_rag_system_prompt_handles_out_of_scope_questions() -> None:
    """Le prompt doit anticiper les questions hors sujet."""
    prompt_lower = RAG_SYSTEM_PROMPT.lower()
    assert "rapport avec les événements" in prompt_lower
    assert "uniquement" in prompt_lower

def test_build_user_prompt_includes_context_and_question() -> None:
    """Le prompt utilisateur doit contenir le contexte et la question."""
    context = "[1] Initiation à l'astronomie — Lanton (2026-01-05)\nSoirée d'observation."
    question = "Quels événements d'astronomie sont disponibles ?"

    user_prompt = build_user_prompt(question, context)

    assert context in user_prompt
    assert question in user_prompt


def test_build_user_prompt_uses_explicit_labels() -> None:
    """Le prompt utilisateur doit utiliser des libellés explicites."""
    user_prompt = build_user_prompt("Une question ?", "Un contexte.")

    assert "Contexte" in user_prompt
    assert "Question" in user_prompt


def test_build_user_prompt_places_context_before_question() -> None:
    """Le contexte doit être placé avant la question pour aider le modèle."""
    user_prompt = build_user_prompt("Une question ?", "Un contexte.")

    assert user_prompt.index("Contexte") < user_prompt.index("Question")
