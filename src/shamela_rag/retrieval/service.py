"""Retrieval service: compose the full pipeline into one ``retrieve`` call (M4-07).

Wiring: translate (EN->AR) -> dense + sparse search -> RRF fusion -> hydrate candidate text and
citation metadata from Postgres -> cross-encoder rerank -> authority boost -> parent/neighbor
expansion. Returns cite-ready ``ExpandedPassage`` objects. Every stage is configurable via
``RetrievalConfig`` (with per-call ``k``/``filters`` overrides).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from shamela_rag.db.models import Book, Chunk, Section
from shamela_rag.retrieval.authority import (
    DEFAULT_PRINTED_BOOST,
    DEFAULT_TRANSCRIPT_PENALTY,
    apply_authority_boost,
)
from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.expand import ContextExpander, ExpandedPassage, ExpansionConfig
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.fusion import (
    ARM_BM25,
    ARM_DENSE,
    ARM_RERANK,
    ARM_ROOT,
    RETRIEVAL_SOURCES_KEY,
    FusedChunk,
    normalize_retrieval_sources,
    reciprocal_rank_fusion,
)
from shamela_rag.retrieval.rerank import RerankCandidate, RerankedChunk, Reranker
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.retrieval.translate import Translator, prepare_retrieval_query


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_limit: int = 50  # top-N per arm (dense, sparse)
    rrf_k: int = 60
    rerank_input_limit: int = 50  # candidates fed to the reranker after fusion
    rerank_top_k: int = 10
    final_k: int = 5  # passages returned after expansion
    translate: bool = True
    order_by_death_hijri: bool = False
    printed_boost: float = DEFAULT_PRINTED_BOOST
    transcript_penalty: float = DEFAULT_TRANSCRIPT_PENALTY
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    use_root_expansion: bool = False


class RetrievalService:
    def __init__(
        self,
        *,
        translator: Translator,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: Reranker,
        expander: ContextExpander,
        session_factory: sessionmaker[Session],
        config: RetrievalConfig | None = None,
        root_retriever: SparseRetriever | None = None,
    ) -> None:
        self._translator = translator
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._reranker = reranker
        self._expander = expander
        self._session_factory = session_factory
        self._config = config or RetrievalConfig()
        self._root = root_retriever

    def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[ExpandedPassage]:
        cfg = config or self._config
        if not question.strip():
            raise ValueError("question must be non-empty")
        query = self._prepare_query(question, cfg)

        dense = self._dense.search(query, limit=cfg.candidate_limit, filters=filters)
        sparse = self._sparse.search(query, limit=cfg.candidate_limit, filters=filters)
        arms: list[tuple[str, Sequence[RetrievedChunk]]] = [
            (ARM_DENSE, dense),
            (ARM_BM25, sparse),
        ]
        if cfg.use_root_expansion and self._root is not None:
            arms.append(
                (ARM_ROOT, self._root.search(query, limit=cfg.candidate_limit, filters=filters))
            )
        fused = reciprocal_rank_fusion(
            [results for _name, results in arms],
            k=cfg.rrf_k,
            limit=cfg.rerank_input_limit,
            arm_names=[name for name, _results in arms],
        )
        candidates = self._hydrate(fused)
        if not candidates:
            return []

        reranked = self._reranker.rerank(query, candidates, top_k=cfg.rerank_top_k)
        if self._reranker.contributes_rerank_source:
            reranked = [_append_rerank_source(chunk) for chunk in reranked]
        boosted = apply_authority_boost(
            reranked,
            printed_boost=cfg.printed_boost,
            transcript_penalty=cfg.transcript_penalty,
            order_by_death_hijri=cfg.order_by_death_hijri,
        )
        final_k = k if k is not None else cfg.final_k
        return self._expander.expand(boosted[:final_k], config=cfg.expansion)

    def _prepare_query(self, question: str, cfg: RetrievalConfig) -> str:
        if cfg.translate:
            return prepare_retrieval_query(question, self._translator).retrieval_text
        return question.strip()

    def _hydrate(self, fused: Sequence[FusedChunk]) -> list[RerankCandidate]:
        """Fetch source text + citation metadata from Postgres, preserving fusion order."""
        if not fused:
            return []
        chunk_ids = [hit.chunk_id for hit in fused]
        sources_by_id = {
            hit.chunk_id: list(hit.payload.get(RETRIEVAL_SOURCES_KEY) or []) for hit in fused
        }
        with self._session_factory() as session:
            rows = session.execute(
                select(Chunk, Book)
                .join(Book, Chunk.book_id == Book.book_id)
                .where(Chunk.id.in_(chunk_ids))
            ).all()
        by_id = {chunk.id: (chunk, book) for chunk, book in rows}

        section_ids = {
            chunk.section_id for chunk, _book in by_id.values() if chunk.section_id is not None
        }
        sections_by_id: dict[int, Section] = {}
        if section_ids:
            with self._session_factory() as session:
                for section in session.execute(
                    select(Section).where(Section.id.in_(section_ids))
                ).scalars():
                    sections_by_id[section.id] = section

        candidates: list[RerankCandidate] = []
        for chunk_id in chunk_ids:
            pair = by_id.get(chunk_id)
            if pair is None:
                continue
            chunk, book = pair
            section: Section | None = (
                sections_by_id.get(chunk.section_id) if chunk.section_id is not None else None
            )
            candidates.append(
                RerankCandidate(
                    chunk_id=chunk.id,
                    text=chunk.source_text,
                    payload={
                        "book_id": chunk.book_id,
                        "category_id": book.category_id,
                        "category_name": book.category_name_ar,
                        "section_id": chunk.section_id,
                        "content_role": chunk.content_role,
                        "page_id": chunk.start_page_id,
                        "book_title": book.title_ar,
                        "author": book.author_name_ar,
                        "author_death_hijri": book.author_death_hijri,
                        "book_type_label": book.book_type_label,
                        "part": chunk.part,
                        "start_page_id": chunk.start_page_id,
                        "end_page_id": chunk.end_page_id,
                        "start_page_num": chunk.start_page_num,
                        "end_page_num": chunk.end_page_num,
                        "section_trail": section.title_trail if section else None,
                        "section_confidence": section.confidence if section else None,
                        RETRIEVAL_SOURCES_KEY: sources_by_id.get(chunk_id, []),
                    },
                )
            )
        return candidates


def _append_rerank_source(chunk: RerankedChunk) -> RerankedChunk:
    existing = chunk.payload.get(RETRIEVAL_SOURCES_KEY) or []
    if not isinstance(existing, list):
        existing = list(existing)
    sources = normalize_retrieval_sources([*existing, ARM_RERANK])
    payload: dict[str, Any] = {**chunk.payload, RETRIEVAL_SOURCES_KEY: sources}
    return RerankedChunk(
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        text=chunk.text,
        payload=payload,
    )
