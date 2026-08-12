from __future__ import annotations

import pytest

from shamela_rag.eval.metrics import (
    aggregate,
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)

_RANKED = [1, 2, 3, 4, 5]
_RELEVANT = {2, 5}


def test_recall_at_k() -> None:
    assert recall_at_k(_RANKED, _RELEVANT, 1) == 0.0
    assert recall_at_k(_RANKED, _RELEVANT, 2) == pytest.approx(0.5)
    assert recall_at_k(_RANKED, _RELEVANT, 5) == pytest.approx(1.0)


def test_recall_empty_relevant_is_zero() -> None:
    assert recall_at_k(_RANKED, set(), 5) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(_RANKED, _RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(_RANKED, _RELEVANT, 5) == pytest.approx(2 / 5)


def test_hit_at_k() -> None:
    assert hit_at_k(_RANKED, _RELEVANT, 1) == 0.0
    assert hit_at_k(_RANKED, _RELEVANT, 2) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(_RANKED, _RELEVANT) == pytest.approx(0.5)  # first relevant at rank 2
    assert reciprocal_rank(_RANKED, set()) == 0.0
    assert reciprocal_rank([9, 8, 7], {7}) == pytest.approx(1 / 3)


def test_dcg_and_ndcg() -> None:
    import math

    # relevant at ranks 2 and 5 -> DCG = 1/log2(3) + 1/log2(6)
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(6)
    assert dcg_at_k(_RANKED, _RELEVANT, 5) == pytest.approx(expected_dcg)

    # ideal: two relevant at ranks 1,2 -> IDCG = 1/log2(2) + 1/log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(_RANKED, _RELEVANT, 5) == pytest.approx(expected_dcg / idcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k([2, 5, 1, 3, 4], {2, 5}, 5) == pytest.approx(1.0)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(_RANKED, _RELEVANT, 0)


def test_aggregate_excludes_adversarial_from_quality() -> None:
    ks = (1, 2)
    labeled = score_query("a", [2, 1], {2}, ks=ks, latency_ms=10.0)  # hit@1 = 1
    missed = score_query("b", [9, 8], {7}, ks=ks, latency_ms=30.0)  # all zero
    adversarial = score_query("c", [1, 2, 3], set(), ks=ks, latency_ms=20.0)

    report = aggregate([labeled, missed, adversarial], ks=ks)

    assert report.num_queries == 3
    assert report.num_adversarial == 1
    # means over the 2 labeled queries only: hit@1 = (1 + 0) / 2
    assert report.hit[1] == pytest.approx(0.5)
    assert report.mrr == pytest.approx((1.0 + 0.0) / 2)
    assert report.mean_latency_ms == pytest.approx((10.0 + 30.0 + 20.0) / 3)
    assert report.p95_latency_ms is not None
