"""Construction des messages du prompt RAG via LangChain.

Ce module utilise LangChain (`ChatPromptTemplate`) pour structurer le prompt
système et le prompt utilisateur du chatbot Écho. Les messages produits par
LangChain sont ensuite convertis en simples dictionnaires
``{"role": ..., "content": ...}`` afin de rester directement compatibles
avec ``client.chat.complete()`` du SDK Mistral.

La recherche FAISS n'est pas touchée : elle reste assurée par la fonction
``search_similar_events()`` dans ``echo_app.indexing.search``.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from echo_app.rag.prompts import RAG_SYSTEM_PROMPT, build_user_prompt


_ROLE_BY_MESSAGE_TYPE = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}


def _message_to_dict(message) -> dict:
    """Convertit un message LangChain en dictionnaire ``{role, content}``."""
    role = _ROLE_BY_MESSAGE_TYPE.get(message.type, "user")
    return {"role": role, "content": message.content}


def build_langchain_messages(question: str, context: str) -> list[dict]:
    """Construit la liste de messages [system, user] pour Mistral via LangChain.

    Le prompt système provient de ``RAG_SYSTEM_PROMPT`` et le prompt
    utilisateur est construit avec ``build_user_prompt(question, context)``.
    Les deux sont injectés dans un ``ChatPromptTemplate`` par variable
    nommée afin d'éviter toute réinterprétation des accolades présentes
    dans le texte du prompt.
    """
    user_prompt = build_user_prompt(question, context)

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}"),
        ]
    )

    messages = template.format_messages(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return [_message_to_dict(message) for message in messages]
