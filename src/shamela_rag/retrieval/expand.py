"""Parent/neighbor context expansion after reranking.

Expands a matched child using Postgres provenance (same ``section_id`` + ``content_role``).
Never crosses sections; body and footnote stay separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from shamela_rag.chunking.tokens import count_tokens
from shamela_rag.db.models import Chunk
from shamela_rag.retrieval.rerank import RerankedChunk
from shamela_rag.retrieval.stable_ids import stable_chunk_id

DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_MAX_EXPANDED_TOKENS = 2048


class ExpandMode(StrEnum):
    NONE = "none"
    NEIGHBORS = "neighbors"
    FULL_SECTION = "full_section"


@dataclass(frozen=True)
class ExpansionConfig:
    mode: ExpandMode = ExpandMode.NEIGHBORS
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW
    max_expanded_tokens: int = DEFAULT_MAX_EXPANDED_TOKENS
    include_context_header: bool = True


@dataclass(frozen=True)
class ExpandedChunkPart:
    chunk_id: int
    source_text: str
    content_role: str
    is_hit: bool


@dataclass(frozen=True)
class ExpandedPassage:
    hit_chunk_id: int
    score: float
    section_id: int | None
    chunk_ids: tuple[int, ...]
    text: str
    parts: tuple[ExpandedChunkPart, ...]
    payload: dict[str, Any] = field(default_factory=dict)


class ChunkNotFoundError(LookupError):
    pass


class ContextExpander:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def expand(
        self,
        hits: Sequence[RerankedChunk],
        *,
        config: ExpansionConfig | None = None,
    ) -> list[ExpandedPassage]:
        cfg = config or ExpansionConfig()
        if cfg.neighbor_window < 0:
            raise ValueError(f"neighbor_window must be >= 0, got {cfg.neighbor_window}")
        if cfg.max_expanded_tokens <= 0:
            raise ValueError(f"max_expanded_tokens must be positive, got {cfg.max_expanded_tokens}")
        if not hits:
            return []

        with self._session_factory() as session:
            return [self._expand_hit(session, hit, cfg) for hit in hits]

    def _expand_hit(
        self, session: Session, hit: RerankedChunk, config: ExpansionConfig
    ) -> ExpandedPassage:
        row = session.get(Chunk, hit.chunk_id)
        if row is None:
            raise ChunkNotFoundError(f"chunk {hit.chunk_id} not found in Postgres")

        if config.mode is ExpandMode.NONE or row.section_id is None:
            return _build_passage(
                session, hit, row, (_chunk_part(row, is_hit=True),), config.include_context_header
            )

        siblings = _section_siblings(session, row.section_id, row.content_role)
        if not siblings:
            return _build_passage(
                session, hit, row, (_chunk_part(row, is_hit=True),), config.include_context_header
            )

        hit_index = _index_of(siblings, hit.chunk_id)
        if config.mode is ExpandMode.NEIGHBORS:
            indices = _neighbor_indices(len(siblings), hit_index, config.neighbor_window)
        else:
            indices = list(range(len(siblings)))

        selected = _apply_token_cap(siblings, hit_index, indices, config.max_expanded_tokens)
        parts = tuple(_chunk_part(siblings[index], is_hit=index == hit_index) for index in selected)
        return _build_passage(session, hit, row, parts, config.include_context_header)


def _section_siblings(session: Session, section_id: int, content_role: str) -> list[Chunk]:
    return list(
        session.execute(
            select(Chunk)
            .where(Chunk.section_id == section_id, Chunk.content_role == content_role)
            .order_by(
                Chunk.start_page_id.asc().nulls_last(),
                Chunk.start_offset.asc().nulls_last(),
                Chunk.id.asc(),
            )
        ).scalars()
    )


def _index_of(chunks: Sequence[Chunk], chunk_id: int) -> int:
    for index, chunk in enumerate(chunks):
        if chunk.id == chunk_id:
            return index
    raise ChunkNotFoundError(f"chunk {chunk_id} missing from its section siblings")


def _neighbor_indices(length: int, hit_index: int, window: int) -> list[int]:
    start = max(0, hit_index - window)
    end = min(length, hit_index + window + 1)
    return list(range(start, end))


def _apply_token_cap(
    chunks: Sequence[Chunk],
    hit_index: int,
    indices: Sequence[int],
    max_tokens: int,
) -> list[int]:
    selected = sorted(set(indices) | {hit_index})

    def total_tokens(chosen: Sequence[int]) -> int:
        return sum(count_tokens(chunks[index].source_text) for index in chosen)

    while len(selected) > 1 and total_tokens(selected) > max_tokens:
        left, right = selected[0], selected[-1]
        if right != hit_index and (left == hit_index or (right - hit_index) >= (hit_index - left)):
            selected.pop()
        else:
            selected.pop(0)
    return selected


def _chunk_part(chunk: Chunk, *, is_hit: bool) -> ExpandedChunkPart:
    return ExpandedChunkPart(
        chunk_id=chunk.id,
        source_text=chunk.source_text,
        content_role=chunk.content_role,
        is_hit=is_hit,
    )


def _build_passage(
    session: Session,
    hit: RerankedChunk,
    hit_row: Chunk,
    parts: Sequence[ExpandedChunkPart],
    include_context_header: bool,
) -> ExpandedPassage:
    body = "\n\n".join(part.source_text for part in parts)
    text = body
    if include_context_header and hit_row.context_header and hit_row.context_header.strip():
        header = hit_row.context_header.strip()
        text = f"{header}\n\n{body}" if body else header

    payload = dict(hit.payload)
    payload.setdefault("book_id", hit_row.book_id)
    payload.setdefault("section_id", hit_row.section_id)
    payload.setdefault("content_role", hit_row.content_role)
    if hit_row.context_header:
        payload.setdefault("context_header", hit_row.context_header)
    if hit_row.start_page_id is not None:
        payload["stable_id"] = stable_chunk_id(session, hit_row)

    return ExpandedPassage(
        hit_chunk_id=hit.chunk_id,
        score=hit.score,
        section_id=hit_row.section_id,
        chunk_ids=tuple(part.chunk_id for part in parts),
        text=text,
        parts=tuple(parts),
        payload=payload,
    )
