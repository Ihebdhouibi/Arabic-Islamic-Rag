"""Deadline enforcement in the retrieval pipeline (issue #147).

The contract requires a hard budget that fails open: once the budget is gone the pipeline stops
starting new expensive work and returns whatever earlier stages produced, saying what it skipped.
These tests drive that with fake arms so no Postgres or Qdrant is needed; ``_hydrate`` is stubbed
because the deadline logic under test sits either side of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shamela_rag.retrieval.expand import ExpandedChunkPart, ExpandedPassage, ExpansionConfig
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.fusion import RETRIEVAL_SOURCES_KEY, FusedChunk
from shamela_rag.retrieval.rerank import RerankCandidate, RerankedChunk
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.retrieval.service import (
    DEGRADED_DEADLINE_HIT,
    DEGRADED_RERANK_SKIPPED,
    DEGRADED_SPARSE_SKIPPED,
    Deadline,
    RetrievalConfig,
    RetrievalService,
)


class _FakeClock:
    """Manual monotonic clock in seconds; ``advance`` moves it forward."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms / 1000.0


class _FakeArm:
    def __init__(self, chunk_ids: Sequence[int]) -> None:
        self._chunk_ids = list(chunk_ids)
        self.calls = 0

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return [
            RetrievedChunk(chunk_id=cid, score=1.0 - i * 0.1, payload={"book_id": 1})
            for i, cid in enumerate(self._chunk_ids)
        ]


class _FakeReranker:
    contributes_rerank_source = True

    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedChunk]:
        self.calls += 1
        # Reverse of fusion order, so a skipped rerank is visibly different downstream.
        chosen = list(reversed(list(candidates)))[: top_k or len(candidates)]
        return [
            RerankedChunk(
                chunk_id=c.chunk_id, score=100.0 - i, text=c.text, payload=dict(c.payload)
            )
            for i, c in enumerate(chosen)
        ]


class _FakeExpander:
    def expand(
        self, hits: Sequence[RerankedChunk], *, config: ExpansionConfig | None = None
    ) -> list[ExpandedPassage]:
        return [
            ExpandedPassage(
                hit_chunk_id=hit.chunk_id,
                score=hit.score,
                section_id=None,
                chunk_ids=(hit.chunk_id,),
                text=hit.text,
                parts=(
                    ExpandedChunkPart(
                        chunk_id=hit.chunk_id,
                        source_text=hit.text,
                        content_role="body",
                        is_hit=True,
                    ),
                ),
                payload=dict(hit.payload),
            )
            for hit in hits
        ]


class _StubTranslator:
    def translate(self, text: str, *, source: Any = None, target: Any = None) -> str:
        return text


class _Service(RetrievalService):
    """Real pipeline, canned hydration (the DB round-trip is not what these tests exercise)."""

    def _hydrate(self, fused: Sequence[FusedChunk]) -> list[RerankCandidate]:
        return [
            RerankCandidate(
                chunk_id=hit.chunk_id,
                text=f"text-{hit.chunk_id}",
                payload={
                    "book_id": 1,
                    "stable_id": f"shamela:1:{hit.chunk_id}:1",
                    RETRIEVAL_SOURCES_KEY: list(hit.payload.get(RETRIEVAL_SOURCES_KEY) or []),
                },
            )
            for hit in fused
        ]


def _build() -> tuple[_Service, _FakeArm, _FakeArm, _FakeReranker]:
    dense = _FakeArm([1, 2, 3])
    sparse = _FakeArm([3, 4, 5])
    reranker = _FakeReranker()
    service = _Service(
        translator=_StubTranslator(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        expander=_FakeExpander(),
        session_factory=None,
        config=RetrievalConfig(
            translate=False,
            final_k=3,
            rerank_top_k=3,
            expansion=ExpansionConfig(),
        ),
    )
    return service, dense, sparse, reranker


def test_fresh_deadline_runs_the_full_pipeline() -> None:
    service, dense, sparse, reranker = _build()
    outcome = service.retrieve_with_outcome("query", deadline=Deadline(5000))

    assert dense.calls == 1
    assert sparse.calls == 1
    assert reranker.calls == 1
    assert outcome.reranked is True
    assert outcome.degraded == ()
    assert outcome.candidates_considered > 0
    assert outcome.passages


def test_expired_deadline_skips_sparse_and_rerank_but_still_returns_results() -> None:
    clock = _FakeClock()
    service, dense, sparse, reranker = _build()
    deadline = Deadline(100, clock=clock)
    clock.advance(150)  # budget already gone before any stage starts

    outcome = service.retrieve_with_outcome("query", deadline=deadline)

    # The dense arm is already paid for; nothing expensive starts after the budget is gone.
    assert dense.calls == 1
    assert sparse.calls == 0
    assert reranker.calls == 0

    # Fails open: partial results, not an empty response or an exception.
    assert outcome.passages
    assert outcome.reranked is False
    assert DEGRADED_SPARSE_SKIPPED in outcome.degraded
    assert DEGRADED_RERANK_SKIPPED in outcome.degraded
    assert DEGRADED_DEADLINE_HIT in outcome.degraded


def test_deadline_expiring_midway_still_skips_rerank() -> None:
    clock = _FakeClock()
    service, dense, sparse, reranker = _build()
    deadline = Deadline(100, clock=clock)

    original_search = sparse.search

    def _slow_search(*args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        clock.advance(500)  # the sparse arm overruns the whole budget
        return original_search(*args, **kwargs)

    sparse.search = _slow_search  # type: ignore[method-assign]
    outcome = service.retrieve_with_outcome("query", deadline=deadline)

    assert sparse.calls == 1  # it ran, it was within budget when it started
    assert reranker.calls == 0  # but the budget was gone by the rerank boundary
    assert outcome.reranked is False
    assert DEGRADED_RERANK_SKIPPED in outcome.degraded
    assert DEGRADED_DEADLINE_HIT in outcome.degraded
    assert outcome.passages


def test_no_deadline_is_unlimited_and_unchanged() -> None:
    service, dense, sparse, reranker = _build()
    outcome = service.retrieve_with_outcome("query")

    assert (dense.calls, sparse.calls, reranker.calls) == (1, 1, 1)
    assert outcome.degraded == ()
    assert outcome.reranked is True


def test_retrieve_still_returns_a_plain_passage_list() -> None:
    service, _dense, _sparse, _reranker = _build()
    passages = service.retrieve("query")

    assert isinstance(passages, list)
    assert passages
    assert all(isinstance(p, ExpandedPassage) for p in passages)


def test_skipped_rerank_preserves_fusion_order() -> None:
    clock = _FakeClock()
    service, _dense, _sparse, _reranker = _build()
    deadline = Deadline(100, clock=clock)
    clock.advance(150)

    degraded_ids = [
        p.hit_chunk_id for p in service.retrieve_with_outcome("q", deadline=deadline).passages
    ]
    ranked_ids = [p.hit_chunk_id for p in service.retrieve_with_outcome("q").passages]

    # The fake reranker reverses order, so the two paths must not agree.
    assert degraded_ids != ranked_ids


def test_deadline_reports_elapsed_time() -> None:
    clock = _FakeClock()
    deadline = Deadline(1000, clock=clock)
    assert deadline.expired is False
    clock.advance(250)
    assert deadline.elapsed_ms == 250
    assert deadline.expired is False
    clock.advance(800)
    assert deadline.expired is True


def test_unlimited_deadline_never_expires() -> None:
    clock = _FakeClock()
    deadline = Deadline(None, clock=clock)
    clock.advance(10_000_000)
    assert deadline.expired is False
