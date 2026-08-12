"""Retrieval service: compose the full pipeline into one ``retrieve`` call (M4-07).

Wiring: translate (EN->AR) -> dense + sparse search -> RRF fusion -> hydrate candidate text and
citation metadata from Postgres -> cross-encoder rerank -> authority boost -> parent/neighbor
expansion. Returns cite-ready ``ExpandedPassage`` objects. Every stage is configurable via
``RetrievalConfig`` (with per-call ``k``/``filters`` overrides).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from shamela_rag.db.models import Book, Chunk
from shamela_rag.retrieval.authority import (
    DEFAULT_PRINTED_BOOST,
    DEFAULT_TRANSCRIPT_PENALTY,
    apply_authority_boost,
)
from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.expand import ContextExpander, ExpandedPassage, ExpansionConfig
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.fusion import reciprocal_rank_fusion
from shamela_rag.retrieval.rerank import RerankCandidate, Reranker
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
    ) -> None:
        self._translator = translator
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._reranker = reranker
        self._expander = expander
        self._session_factory = session_factory
        self._config = config or RetrievalConfig()

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
        fused = reciprocal_rank_fusion([dense, sparse], k=cfg.rrf_k, limit=cfg.rerank_input_limit)
        candidates = self._hydrate([hit.chunk_id for hit in fused])
        if not candidates:
            return []

        reranked = self._reranker.rerank(query, candidates, top_k=cfg.rerank_top_k)
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

    def _hydrate(self, chunk_ids: Sequence[int]) -> list[RerankCandidate]:
        """Fetch source text + citation metadata from Postgres, preserving fusion order."""
        if not chunk_ids:
            return []
        with self._session_factory() as session:
            rows = session.execute(
                select(Chunk, Book)
                .join(Book, Chunk.book_id == Book.book_id)
                .where(Chunk.id.in_(chunk_ids))
            ).all()
        by_id = {chunk.id: (chunk, book) for chunk, book in rows}

        candidates: list[RerankCandidate] = []
        for chunk_id in chunk_ids:
            pair = by_id.get(chunk_id)
            if pair is None:
                continue
            chunk, book = pair
            candidates.append(
                RerankCandidate(
                    chunk_id=chunk.id,
                    text=chunk.source_text,
                    payload={
                        "book_id": chunk.book_id,
                        "category_id": book.category_id,
                        "section_id": chunk.section_id,
                        "content_role": chunk.content_role,
                        "page_id": chunk.start_page_id,
                        "book_title": book.title_ar,
                        "author": book.author_name_ar,
                        "author_death_hijri": book.author_death_hijri,
                        "book_type_label": book.book_type_label,
                    },
                )
            )
        return candidates
