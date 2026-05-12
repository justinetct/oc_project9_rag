"""Service RAG d'Écho : recherche FAISS + génération Mistral.

Ce module orchestre la chaîne RAG :
- recherche sémantique des chunks d'événements via search_similar_events() ;
- construction d'un contexte numéroté à partir des chunks retrouvés ;
- construction des messages du prompt via LangChain ;
- appel au modèle Mistral pour générer la réponse ;
- extraction des sources affichables, dédoublonnées par event_id.

La recherche FAISS n'est jamais réimplémentée ici : RagService réutilise la
fonction search_similar_events() existante dans echo_app.indexing.search.

Les prompts métier sont centralisés dans echo_app.rag.prompts, et leur
assemblage en messages [system, user] passe par echo_app.rag.langchain_chain.
"""

from __future__ import annotations

from echo_app.config import MISTRAL_MODEL
from echo_app.indexing.embeddings import get_mistral_client
from echo_app.indexing.search import search_similar_events
from echo_app.rag.langchain_chain import build_langchain_messages
from src.config import SEED


DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2
CHUNK_SEPARATOR = "\n---\n"
EMPTY_RESULT_ANSWER = (
    "Je n'ai trouvé aucun événement pertinent pour votre question."
)


def _format_chunk_header(rank: int, metadata: dict) -> str:
    """Retourne une en-tête lisible pour un chunk : [n] titre — ville (date)."""
    title = metadata.get("title") or "(sans titre)"
    city = metadata.get("city") or "-"
    start_date = metadata.get("start_date") or ""
    if start_date:
        return f"[{rank}] {title} — {city} ({start_date})"
    return f"[{rank}] {title} — {city}"


def build_context(results: list[dict]) -> str:
    """Construit le bloc de contexte à injecter dans le prompt utilisateur."""
    blocks: list[str] = []
    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        header = _format_chunk_header(rank, metadata)
        text = (result.get("text") or "").strip()
        blocks.append(f"{header}\n{text}")
    return CHUNK_SEPARATOR.join(blocks)


def extract_sources(results: list[dict]) -> list[dict]:
    """Extrait la liste des sources affichables, dédoublonnée par event_id.

    L'ordre des résultats est préservé : la première occurrence d'un event_id
    est conservée, les chunks suivants du même événement sont ignorés.
    """
    seen_source_ids: set[str] = set()
    sources: list[dict] = []
    for result in results:
        source_id = str(result.get("event_id") or result.get("chunk_id"))
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        metadata = result.get("metadata") or {}
        sources.append(
            {
                "event_id": result.get("event_id"),
                "title": metadata.get("title"),
                "city": metadata.get("city"),
                "start_date": metadata.get("start_date"),
                "url": metadata.get("url"),
            }
        )
    return sources


class RagService:
    """Orchestre la chaîne recherche + génération pour le chatbot Écho."""

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif.")
        self.top_k = top_k
        self.model = MISTRAL_MODEL

    def ask(self, question: str) -> dict:
        """Répond à une question utilisateur via la chaîne RAG complète."""
        if not question or not str(question).strip():
            raise ValueError("La question est vide.")

        results = search_similar_events(question, top_k=self.top_k)

        if not results:
            return {
                "question": question,
                "answer": EMPTY_RESULT_ANSWER,
                "sources": [],
            }

        context = build_context(results)
        messages = build_langchain_messages(question=question, context=context)
        answer = self._generate_answer(messages)

        return {
            "question": question,
            "answer": answer,
            "sources": extract_sources(results),
        }

    def _generate_answer(self, messages: list[dict]) -> str:
        """Appelle Mistral pour générer la réponse à partir des messages.

        Le client Mistral est créé uniquement au moment de générer la réponse,
        ce qui permet d'instancier RagService dans les tests sans clé API.
        """
        client = get_mistral_client()
        response = client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            random_seed=SEED,
        )
        return response.choices[0].message.content or ""
