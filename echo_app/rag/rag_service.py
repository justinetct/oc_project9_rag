"""Service RAG d'Écho : retriever FAISS LangChain + génération Mistral.

Ce module orchestre la chaîne RAG :

- récupération des chunks d'événements pertinents via le retriever
  LangChain obtenu avec ``vectorstore.as_retriever()`` ;
- filtrage des chunks par écart de distance L2 entre sources
  (``filter_documents_by_l2_gap``) : on écarte les sources nettement
  plus lointaines que la meilleure ;
- construction d'un contexte numéroté à partir des Documents retenus ;
- construction des messages du prompt via ``ChatPromptTemplate`` ;
- appel au modèle Mistral pour générer la réponse ;
- extraction des sources affichables, dédoublonnées par event_id.

Le vector store est le FAISS standard de LangChain
(``langchain_community.vectorstores.FAISS``). Il est rechargé une seule
fois (lazy) puis exposé en tant que retriever via
``.as_retriever(search_kwargs={"k": top_k})``.

Les prompts métier sont centralisés dans ``echo_app.rag.prompts`` et leur
assemblage en messages [system, user] passe par
``echo_app.rag.langchain_chain``.
"""

from __future__ import annotations

import time

from langchain_core.documents import Document

from echo_app.config import MISTRAL_MODEL
from echo_app.indexing.embeddings import get_mistral_client
from echo_app.indexing.langchain_embeddings import MistralLangChainEmbeddings
from echo_app.indexing.langchain_faiss_store import load_langchain_vector_store
from echo_app.rag.langchain_chain import build_langchain_messages
from src.config import SEED


DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2
# Au-delà de cet écart de distance L2 entre deux sources consécutives,
# la source la plus lointaine (et les suivantes) est écartée du contexte.
L2_GAP_THRESHOLD = 0.10
CHUNK_SEPARATOR = "\n---\n"
EMPTY_RESULT_ANSWER = (
    "Je n'ai trouvé aucun événement pertinent pour votre question."
)
MISTRAL_RETRY_DELAY_SECONDS = 2


def _is_mistral_capacity_error(exc: BaseException) -> bool:
    """Détecte un échec Mistral lié à la capacité du tier.

    Mistral renvoie un HTTP 429 avec un message
    ``service_tier_capacity_exceeded`` lorsque le modèle est temporairement
    saturé côté serveur. On reconnaît aussi ``Status 429`` et
    ``capacity exceeded`` pour rester tolérant aux variations de format
    selon les versions du SDK.
    """
    message = str(exc).lower()
    return (
        "status 429" in message
        or "service_tier_capacity_exceeded" in message
        or "capacity exceeded" in message
    )


def _format_chunk_header(rank: int, metadata: dict) -> str:
    """Retourne une en-tête lisible pour un chunk : [n] titre — ville (date)."""
    title = metadata.get("title") or "(sans titre)"
    city = metadata.get("city") or "-"
    start_date = metadata.get("start_date") or ""
    if start_date:
        return f"[{rank}] {title} — {city} ({start_date})"
    return f"[{rank}] {title} — {city}"


def build_context(documents: list[Document]) -> str:
    """Construit le bloc de contexte à injecter dans le prompt utilisateur.

    Chaque Document retrouvé par le retriever produit un bloc numéroté
    [n] précédé d'une en-tête lisible, séparé du suivant par ``\\n---\\n``.
    """
    blocks: list[str] = []
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        header = _format_chunk_header(rank, metadata)
        text = (document.page_content or "").strip()
        blocks.append(f"{header}\n{text}")
    return CHUNK_SEPARATOR.join(blocks)


def extract_sources(documents: list[Document]) -> list[dict]:
    """Extrait la liste des sources affichables, dédoublonnée par event_id.

    L'ordre des documents est préservé : la première occurrence d'un
    event_id est conservée, les chunks suivants du même événement sont
    ignorés.
    """
    seen_source_ids: set[str] = set()
    sources: list[dict] = []
    for document in documents:
        metadata = document.metadata or {}
        source_id = str(metadata.get("event_id") or metadata.get("chunk_id"))
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        sources.append(
            {
                "event_id": metadata.get("event_id"),
                "title": metadata.get("title"),
                "city": metadata.get("city"),
                "start_date": metadata.get("start_date"),
                "url": metadata.get("url"),
            }
        )
    return sources


def filter_documents_by_l2_gap(
    documents: list[Document],
    scored: list[tuple[Document, float]],
    threshold: float = L2_GAP_THRESHOLD,
) -> list[Document]:
    """Filtre les chunks récupérés selon l'écart de distance L2 entre sources.

    ``documents`` est la liste de chunks renvoyée par le retriever LangChain.
    ``scored`` est la liste ``(Document, distance L2)`` issue de
    ``similarity_search_with_score`` pour la même question : le retriever
    n'exposant pas les distances, cette seconde requête sert uniquement à
    les récupérer. Les chunks sont dédoublonnés en « sources » (1er chunk
    par event_id) ; au premier écart de distance L2 supérieur à
    ``threshold``, cette source et les suivantes sont écartées. On renvoie
    les chunks de ``documents`` appartenant aux sources gardées, dans
    l'ordre d'origine.
    """
    if not documents:
        return []

    # Distance L2 par event_id (1re occurrence dans le résultat scoré).
    distance_by_event: dict[str, float] = {}
    for document, distance in scored:
        event_id = str((document.metadata or {}).get("event_id"))
        if event_id not in distance_by_event:
            distance_by_event[event_id] = distance

    # Sources = 1er chunk par event_id, dans l'ordre du retriever.
    source_event_ids: list[str] = []
    seen_event_ids: set[str] = set()
    for document in documents:
        event_id = str((document.metadata or {}).get("event_id"))
        if event_id not in seen_event_ids:
            seen_event_ids.add(event_id)
            source_event_ids.append(event_id)

    # On garde la source la plus proche, puis on s'arrête au 1er écart trop grand.
    kept_event_ids = {source_event_ids[0]}
    previous_distance = distance_by_event.get(source_event_ids[0], 0.0)
    for event_id in source_event_ids[1:]:
        distance = distance_by_event.get(event_id)
        if distance is None or distance - previous_distance > threshold:
            break
        kept_event_ids.add(event_id)
        previous_distance = distance

    return [
        document
        for document in documents
        if str((document.metadata or {}).get("event_id")) in kept_event_ids
    ]


class RagService:
    """Orchestre la chaîne retriever LangChain + génération Mistral."""

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif.")
        self.top_k = top_k
        self.model = MISTRAL_MODEL
        self._retriever = None

    @property
    def retriever(self):
        """Retriever LangChain construit à partir du vector store FAISS.

        Le retriever est créé une seule fois, au premier appel : on charge
        le vector store FAISS LangChain depuis le disque et on en extrait
        un retriever via ``.as_retriever(search_kwargs={"k": top_k})``.
        Le chargement est différé pour permettre aux tests d'injecter un
        faux retriever via ``service._retriever = fake_retriever``.
        """
        if self._retriever is None:
            embeddings = MistralLangChainEmbeddings()
            vectorstore = load_langchain_vector_store(embeddings)
            self._retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.top_k},
            )
        return self._retriever

    def reset_retriever_cache(self) -> None:
        """Invalide le retriever en cache pour forcer un rechargement.

        À appeler après une reconstruction de l'index (via l'API
        ``POST /rebuild``) afin que le prochain ``ask()`` reconstruise un
        retriever à partir du nouveau vector store.
        """
        self._retriever = None

    def ask(self, question: str, *, include_contexts: bool = False) -> dict:
        """Répond à une question utilisateur via la chaîne RAG complète.

        Récupère les chunks via FAISS, applique le filtre d'écart L2
        (``filter_documents_by_l2_gap``), génère la réponse avec Mistral
        et renvoie ``{question, answer, sources}``.

        Le paramètre ``include_contexts`` est réservé à l'évaluation
        (script ``08_evaluate_rag.py``) : quand il vaut ``True``, le dict
        retourné contient en plus une clé ``contexts`` listant le
        ``page_content`` des chunks retenus, sous la forme attendue par
        Ragas.
        """
        if not question or not str(question).strip():
            raise ValueError("La question est vide.")

        documents = self.retriever.invoke(question)

        if not documents:
            result = {
                "question": question,
                "answer": EMPTY_RESULT_ANSWER,
                "sources": [],
            }
            if include_contexts:
                result["contexts"] = []
            return result

        # Le retriever LangChain n'expose pas les distances : on relance la
        # même recherche FAISS avec les scores, uniquement pour le filtre.
        scored = self.retriever.vectorstore.similarity_search_with_score(
            question, k=self.top_k
        )
        documents = filter_documents_by_l2_gap(documents, scored)

        context = build_context(documents)
        messages = build_langchain_messages(question=question, context=context)
        answer = self._generate_answer(messages)

        result = {
            "question": question,
            "answer": answer,
            "sources": extract_sources(documents),
        }
        if include_contexts:
            result["contexts"] = [doc.page_content or "" for doc in documents]
        return result

    def _generate_answer(self, messages: list[dict]) -> str:
        """Appelle Mistral pour générer la réponse à partir des messages.

        Le client Mistral est créé uniquement au moment de générer la
        réponse, ce qui permet d'instancier RagService dans les tests
        sans clé API.

        Si Mistral renvoie un HTTP 429 ``service_tier_capacity_exceeded``
        (le modèle est temporairement saturé côté serveur), on attend
        ``MISTRAL_RETRY_DELAY_SECONDS`` secondes puis on retente une
        seule fois. Toute autre exception remonte immédiatement pour
        être loggée par l'API.
        """
        client = get_mistral_client()
        try:
            return self._call_chat_complete(client, messages)
        except Exception as exc:
            if not _is_mistral_capacity_error(exc):
                raise
            time.sleep(MISTRAL_RETRY_DELAY_SECONDS)
            return self._call_chat_complete(client, messages)

    def _call_chat_complete(self, client, messages: list[dict]) -> str:
        """Exécute un appel unique à ``client.chat.complete()`` Mistral."""
        response = client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            random_seed=SEED,
        )
        return response.choices[0].message.content or ""
