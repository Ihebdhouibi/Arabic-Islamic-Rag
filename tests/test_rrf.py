from __future__ import annotations

import pytest

from shamela_rag.retrieval.fusion import reciprocal_rank_fusion
from shamela_rag.retrieval.results import RetrievedChunk


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, score=0.0, payload={"book_id": chunk_id})


def test_rrf_fuses_two_ranked_lists() -> None:
    dense = [_chunk(1), _chunk(2), _chunk(3)]
    sparse = [_chunk(1), _chunk(4), _chunk(2)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert [c.chunk_id for c in fused] == [1, 2, 4, 3]
    assert fused[0].score == pytest.approx(2.0 / 61.0)


def test_rrf_rewards_ranking_in_both_lists() -> None:
    # id 2 is moderate in both; id 4 is high in only one -> 2 outranks 4.
    dense = [_chunk(9), _chunk(2)]
    sparse = [_chunk(4), _chunk(2)]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0].chunk_id == 2


def test_rrf_single_list_preserves_order() -> None:
    fused = reciprocal_rank_fusion([[_chunk(5), _chunk(6), _chunk(7)]])
    assert [c.chunk_id for c in fused] == [5, 6, 7]


def test_rrf_ties_break_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([[_chunk(2)], [_chunk(1)]])
    assert [c.chunk_id for c in fused] == [1, 2]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_limit_truncates() -> None:
    dense = [_chunk(1), _chunk(2), _chunk(3)]
    fused = reciprocal_rank_fusion([dense], limit=2)
    assert [c.chunk_id for c in fused] == [1, 2]


def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_preserves_first_seen_payload() -> None:
    dense = [RetrievedChunk(chunk_id=1, score=0.0, payload={"book_id": 10})]
    sparse = [RetrievedChunk(chunk_id=1, score=0.0, payload={"book_id": 99})]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].payload == {"book_id": 10}


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([[_chunk(1)]], k=0)
