"""Authority boost and optional era ordering after rerank (M4-05).

Printed works (``كتاب``) are boosted over transcribed lessons (``دروس مفرغة``). When
``order_by_death_hijri`` is enabled (debate-history questions), ties prefer earlier authors.
The death-year sentinel ``99999`` is treated as unknown (same as the rest of the codebase).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shamela_rag.chunking.context_header import normalize_death_year
from shamela_rag.retrieval.rerank import RerankedChunk

PRINTED_BOOK_TYPE = "كتاب"
TRANSCRIPT_BOOK_TYPE = "دروس مفرغة"
DEFAULT_PRINTED_BOOST = 0.1
DEFAULT_TRANSCRIPT_PENALTY = 0.1
_UNKNOWN_ERA = 10**9


def _book_type_label(payload: dict[str, Any]) -> str | None:
    value = payload.get("book_type_label")
    return value if isinstance(value, str) and value.strip() else None


def _author_death_hijri(payload: dict[str, Any]) -> int | None:
    value = payload.get("author_death_hijri")
    if not isinstance(value, int):
        return None
    return normalize_death_year(value)


def adjusted_authority_score(
    score: float,
    book_type_label: str | None,
    *,
    printed_boost: float = DEFAULT_PRINTED_BOOST,
    transcript_penalty: float = DEFAULT_TRANSCRIPT_PENALTY,
) -> float:
    if book_type_label == PRINTED_BOOK_TYPE:
        return score + printed_boost
    if book_type_label == TRANSCRIPT_BOOK_TYPE:
        return score - transcript_penalty
    return score


def apply_authority_boost(
    chunks: Sequence[RerankedChunk],
    *,
    printed_boost: float = DEFAULT_PRINTED_BOOST,
    transcript_penalty: float = DEFAULT_TRANSCRIPT_PENALTY,
    order_by_death_hijri: bool = False,
) -> list[RerankedChunk]:
    """Re-score and re-order reranked chunks by authority (+ optional death_hijri)."""
    if not chunks:
        return []

    adjusted: list[RerankedChunk] = []
    for chunk in chunks:
        new_score = adjusted_authority_score(
            chunk.score,
            _book_type_label(chunk.payload),
            printed_boost=printed_boost,
            transcript_penalty=transcript_penalty,
        )
        adjusted.append(
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                score=new_score,
                text=chunk.text,
                payload=chunk.payload,
            )
        )

    def sort_key(chunk: RerankedChunk) -> tuple[float, int, int]:
        death = _author_death_hijri(chunk.payload)
        era = death if order_by_death_hijri and death is not None else _UNKNOWN_ERA
        # Higher score first; for debate-history, earlier death_hijri next; then chunk_id.
        return (-chunk.score, era if order_by_death_hijri else 0, chunk.chunk_id)

    return sorted(adjusted, key=sort_key)
