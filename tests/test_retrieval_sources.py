from __future__ import annotations

import pytest

from shamela_rag.retrieval.fusion import (
    ARM_BM25,
    ARM_DENSE,
    ARM_RERANK,
    ARM_ROOT,
    RETRIEVAL_SOURCES_KEY,
    normalize_retrieval_sources,
    reciprocal_rank_fusion,
)
from shamela_rag.retrieval.rerank import LexicalOverlapReranker, RerankCandidate, RerankedChunk
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.retrieval.service import _append_rerank_source


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, score=0.0, payload={"book_id": chunk_id})


def test_normalize_retrieval_sources_orders_known_arms() -> None:
    assert normalize_retrieval_sources([ARM_RERANK, ARM_ROOT, ARM_DENSE, ARM_BM25]) == [
        ARM_DENSE,
        ARM_BM25,
        ARM_ROOT,
        ARM_RERANK,
    ]


def test_rrf_tags_arms_for_shared_and_unique_hits() -> None:
    dense = [_chunk(1), _chunk(2)]
    sparse = [_chunk(1), _chunk(3)]

    fused = reciprocal_rank_fusion(
        [dense, sparse],
        arm_names=[ARM_DENSE, ARM_BM25],
    )
    by_id = {hit.chunk_id: hit.payload[RETRIEVAL_SOURCES_KEY] for hit in fused}

    assert by_id[1] == [ARM_DENSE, ARM_BM25]
    assert by_id[2] == [ARM_DENSE]
    assert by_id[3] == [ARM_BM25]


def test_rrf_tags_root_arm_when_named() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk(7)], [_chunk(7)], [_chunk(8)]],
        arm_names=[ARM_DENSE, ARM_BM25, ARM_ROOT],
    )
    by_id = {hit.chunk_id: hit.payload[RETRIEVAL_SOURCES_KEY] for hit in fused}
    assert by_id[7] == [ARM_DENSE, ARM_BM25]
    assert by_id[8] == [ARM_ROOT]


def test_rrf_without_arm_names_omits_retrieval_sources() -> None:
    fused = reciprocal_rank_fusion([[_chunk(1)], [_chunk(1)]])
    assert RETRIEVAL_SOURCES_KEY not in fused[0].payload


def test_rrf_rejects_mismatched_arm_names() -> None:
    with pytest.raises(ValueError, match="arm_names length"):
        reciprocal_rank_fusion([[_chunk(1)]], arm_names=[ARM_DENSE, ARM_BM25])


def test_append_rerank_source_adds_rerank_once() -> None:
    chunk = RerankedChunk(
        chunk_id=1,
        score=1.0,
        text="x",
        payload={RETRIEVAL_SOURCES_KEY: [ARM_DENSE, ARM_BM25]},
    )
    updated = _append_rerank_source(chunk)
    assert updated.payload[RETRIEVAL_SOURCES_KEY] == [ARM_DENSE, ARM_BM25, ARM_RERANK]
    again = _append_rerank_source(updated)
    assert again.payload[RETRIEVAL_SOURCES_KEY] == [ARM_DENSE, ARM_BM25, ARM_RERANK]


def test_lexical_rerank_preserves_sources_without_rerank_tag() -> None:
    candidates = [
        RerankCandidate(
            chunk_id=1,
            text=" overlapping tokens here ",
            payload={RETRIEVAL_SOURCES_KEY: [ARM_DENSE]},
        ),
        RerankCandidate(
            chunk_id=2,
            text="unrelated",
            payload={RETRIEVAL_SOURCES_KEY: [ARM_BM25]},
        ),
    ]
    ranked = LexicalOverlapReranker().rerank("overlapping tokens", candidates, top_k=2)
    assert ranked[0].payload[RETRIEVAL_SOURCES_KEY] == [ARM_DENSE]
    assert ARM_RERANK not in ranked[0].payload[RETRIEVAL_SOURCES_KEY]
