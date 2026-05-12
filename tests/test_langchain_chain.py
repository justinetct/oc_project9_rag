"""Tests unitaires pour la construction des messages via LangChain.

Ces tests vérifient que ``build_langchain_messages`` retourne un format
``[{role, content}, ...]`` compatible avec ``client.chat.complete()``,
en s'appuyant sur les prompts centralisés dans ``echo_app.rag.prompts``.
Aucun appel réseau n'est effectué.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from echo_app.rag.langchain_chain import build_langchain_messages  # noqa: E402
from echo_app.rag.prompts import RAG_SYSTEM_PROMPT  # noqa: E402


CONTEXT = "[1] Initiation à l'astronomie — Lanton (2026-01-05)\nSoirée d'observation du ciel."
QUESTION = "Quels événements d'astronomie sont disponibles ?"


def test_build_langchain_messages_returns_two_messages() -> None:
    """La fonction doit retourner exactement deux messages : system puis user."""
    messages = build_langchain_messages(question=QUESTION, context=CONTEXT)

    assert isinstance(messages, list)
    assert len(messages) == 2


def test_build_langchain_messages_first_message_is_system() -> None:
    """Le premier message doit avoir le rôle ``system``."""
    messages = build_langchain_messages(question=QUESTION, context=CONTEXT)

    assert messages[0]["role"] == "system"


def test_build_langchain_messages_second_message_is_user() -> None:
    """Le second message doit avoir le rôle ``user``."""
    messages = build_langchain_messages(question=QUESTION, context=CONTEXT)

    assert messages[1]["role"] == "user"


def test_build_langchain_messages_system_content_matches_prompt() -> None:
    """Le contenu système doit reprendre le prompt centralisé."""
    messages = build_langchain_messages(question=QUESTION, context=CONTEXT)

    system_content = messages[0]["content"]
    assert "Écho" in system_content
    assert "Bassin d'Arcachon" in system_content
    assert system_content == RAG_SYSTEM_PROMPT


def test_build_langchain_messages_user_content_includes_context_and_question() -> None:
    """Le contenu utilisateur doit contenir le contexte et la question."""
    messages = build_langchain_messages(question=QUESTION, context=CONTEXT)

    user_content = messages[1]["content"]
    assert CONTEXT in user_content
    assert QUESTION in user_content
    assert "Contexte" in user_content
    assert "Question" in user_content


def test_build_langchain_messages_format_is_mistral_compatible() -> None:
    """Chaque message doit être un dict avec uniquement ``role`` et ``content``."""
    messages = build_langchain_messages(question=QUESTION, context=CONTEXT)

    for message in messages:
        assert isinstance(message, dict)
        assert set(message.keys()) == {"role", "content"}
        assert isinstance(message["role"], str)
        assert isinstance(message["content"], str)
