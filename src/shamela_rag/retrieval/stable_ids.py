"""Stable citation ids: ``shamela:{book_id}:{page_id}:{ordinal}`` (survive re-ingest)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from shamela_rag.db.models import Chunk

STABLE_ID_PREFIX = "shamela"
_STABLE_ID_PATTERN = re.compile(
    rf"^{STABLE_ID_PREFIX}:(?P<book_id>\d+):(?P<page_id>\d+):(?P<ordinal>\d+)$"
)


@dataclass(frozen=True, slots=True)
class StableChunkKey:
    book_id: int
    page_id: int
    ordinal: int


def format_stable_chunk_id(book_id: int, page_id: int, ordinal: int) -> str:
    if book_id <= 0 or page_id <= 0 or ordinal <= 0:
        raise ValueError(
            f"book_id, page_id, and ordinal must be positive, got {book_id}, {page_id}, {ordinal}"
        )
    return f"{STABLE_ID_PREFIX}:{book_id}:{page_id}:{ordinal}"


def parse_stable_chunk_id(stable_id: str) -> StableChunkKey:
    match = _STABLE_ID_PATTERN.fullmatch(stable_id.strip())
    if match is None:
        raise ValueError(f"invalid stable chunk id: {stable_id!r}")
    return StableChunkKey(
        book_id=int(match.group("book_id")),
        page_id=int(match.group("page_id")),
        ordinal=int(match.group("ordinal")),
    )


def _chunks_on_page(session: Session, book_id: int, page_id: int) -> list[Chunk]:
    return list(
        session.execute(
            select(Chunk)
            .where(Chunk.book_id == book_id, Chunk.start_page_id == page_id)
            .order_by(
                Chunk.start_offset.asc().nulls_last(),
                Chunk.content_role.asc(),
                Chunk.id.asc(),
            )
        ).scalars()
    )


def page_ordinal(session: Session, chunk: Chunk) -> int:
    if chunk.start_page_id is None:
        return 1
    siblings = _chunks_on_page(session, chunk.book_id, chunk.start_page_id)
    for index, sibling in enumerate(siblings, start=1):
        if sibling.id == chunk.id:
            return index
    raise LookupError(f"chunk {chunk.id} missing from its page siblings")


def stable_chunk_id(session: Session, chunk: Chunk) -> str:
    if chunk.start_page_id is None:
        raise ValueError(f"chunk {chunk.id} has no start_page_id for stable id")
    return format_stable_chunk_id(chunk.book_id, chunk.start_page_id, page_ordinal(session, chunk))


def resolve_stable_chunk_id(session: Session, stable_id: str) -> Chunk | None:
    key = parse_stable_chunk_id(stable_id)
    siblings = _chunks_on_page(session, key.book_id, key.page_id)
    if key.ordinal < 1 or key.ordinal > len(siblings):
        return None
    return siblings[key.ordinal - 1]
