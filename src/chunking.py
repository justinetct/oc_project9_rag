"""Fonctions simples de chunking pour les documents OpenAgenda."""

from __future__ import annotations

from copy import deepcopy

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE
from src.preprocessing import normalize_text


def _validate_chunking_params(chunk_size: int, chunk_overlap: int) -> None:
    """Vérifie que les paramètres de chunking sont cohérents."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _get_min_chunk_size(chunk_size: int) -> int:
    """Adapte la taille minimale au chunk_size utilisé."""
    return min(MIN_CHUNK_SIZE, max(chunk_size // 2, 1))


def _find_chunk_end(text: str, start: int, max_end: int, min_chunk_size: int) -> int:
    """Cherche une coupure simple près de la fin de la fenêtre."""
    if max_end >= len(text):
        return len(text)

    min_end = min(start + min_chunk_size, max_end)

    for separator in ["\n\n", "\n", " "]:
        split_at = text.rfind(separator, min_end, max_end + 1)
        if split_at != -1:
            if separator == " ":
                return split_at
            return split_at + len(separator)

    return max_end


def _merge_small_chunks(chunks: list[str], min_chunk_size: int) -> list[str]:
    """Fusionne les morceaux finaux trop courts avec le chunk précédent."""
    merged_chunks: list[str] = []

    for chunk in chunks:
        if len(chunk) < min_chunk_size and merged_chunks:
            merged_chunks[-1] = f"{merged_chunks[-1].rstrip()}\n\n{chunk.lstrip()}".strip()
            continue
        merged_chunks.append(chunk)

    return [chunk for chunk in merged_chunks if chunk.strip()]


def _find_next_start(text: str, proposed_start: int) -> int:
    """Décale légèrement le prochain départ pour éviter les débuts de mot tronqués."""
    if proposed_start <= 0 or proposed_start >= len(text):
        return proposed_start

    if text[proposed_start - 1].isspace() or text[proposed_start].isspace():
        return proposed_start

    search_limit = min(proposed_start + 80, len(text))

    for separator in ["\n\n", "\n", " "]:
        next_separator = text.find(separator, proposed_start, search_limit)
        if next_separator != -1:
            return next_separator + len(separator)

    return proposed_start


def _extract_title(document: dict) -> str:
    """Récupère un titre simple depuis les métadonnées ou le texte."""
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        title = normalize_text(metadata.get("title"))
        if title:
            return title

    document_text = str(document.get("document_text") or "").strip()
    first_line = document_text.splitlines()[0].strip() if document_text else ""
    if first_line.startswith("# "):
        return normalize_text(first_line[2:])

    return ""


def _prepend_title(chunk_text: str, title: str) -> str:
    """Ajoute un titre au chunk si nécessaire."""
    if not title:
        return chunk_text

    title_heading = f"# {title}"
    if chunk_text.startswith(title_heading):
        return chunk_text

    return f"{title_heading}\n\n{chunk_text}".strip()


def chunk_long_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Découpe un texte long en morceaux avec overlap léger."""
    _validate_chunking_params(chunk_size, chunk_overlap)
    min_chunk_size = _get_min_chunk_size(chunk_size)

    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        max_end = min(start + chunk_size, text_length)
        end = _find_chunk_end(cleaned_text, start, max_end, min_chunk_size)

        if end <= start:
            end = max_end

        chunk_text = cleaned_text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, 0)
        next_start = _find_next_start(cleaned_text, next_start)
        if next_start <= start:
            next_start = end
        start = next_start

    return _merge_small_chunks(chunks, min_chunk_size)


def build_chunks(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Construit les chunks à partir des documents OpenAgenda."""
    chunks: list[dict] = []

    for document in documents:
        event_id = normalize_text(document.get("event_id"))
        document_text = str(document.get("document_text") or "").strip()
        metadata = document.get("metadata")

        if not event_id or not document_text:
            continue
        if not isinstance(metadata, dict):
            metadata = {}

        document_chunks = chunk_long_text(
            document_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not document_chunks:
            continue

        title = _extract_title(document)
        chunk_count = len(document_chunks)

        for chunk_index, chunk_text in enumerate(document_chunks):
            final_chunk_text = chunk_text
            if chunk_count > 1:
                final_chunk_text = _prepend_title(chunk_text, title)

            chunks.append(
                {
                    "chunk_id": f"{event_id}_{chunk_index}",
                    "event_id": event_id,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "chunk_text": final_chunk_text,
                    "metadata": deepcopy(metadata),
                }
            )

    return chunks


def get_chunk_stats(chunks: list[dict]) -> dict:
    """Retourne quelques statistiques simples sur les chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_events": 0,
            "single_chunk_events": 0,
            "multi_chunk_events": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "mean_chunk_length": 0.0,
            "mean_chunks_per_event": 0.0,
            "empty_chunk_count": 0,
        }

    chunk_lengths = [len(str(chunk.get("chunk_text") or "")) for chunk in chunks]

    chunks_per_event: dict[str, int] = {}
    for chunk in chunks:
        event_id = normalize_text(chunk.get("event_id"))
        if event_id:
            chunks_per_event[event_id] = chunks_per_event.get(event_id, 0) + 1

    single_chunk_events = sum(1 for count in chunks_per_event.values() if count == 1)
    multi_chunk_events = sum(1 for count in chunks_per_event.values() if count > 1)

    return {
        "total_chunks": len(chunks),
        "total_events": len(chunks_per_event),
        "single_chunk_events": single_chunk_events,
        "multi_chunk_events": multi_chunk_events,
        "min_chunk_length": min(chunk_lengths),
        "max_chunk_length": max(chunk_lengths),
        "mean_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 1),
        "mean_chunks_per_event": round(len(chunks) / len(chunks_per_event), 1),
        "empty_chunk_count": sum(1 for length in chunk_lengths if length == 0),
    }
