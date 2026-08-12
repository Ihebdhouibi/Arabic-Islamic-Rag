"""Retrieval metric math (M6-02): recall@k, hit@k, MRR, nDCG@k over ranked results.

All functions operate on a ranked sequence of hashable keys (e.g. book ids) and a set of relevant
keys, so they are pure and independent of the retrieval backend. Relevance is binary.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass


def _top_k(ranked: Sequence[Hashable], k: int) -> list[Hashable]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return list(ranked[:k])


def recall_at_k(ranked: Sequence[Hashable], relevant: AbstractSet[Hashable], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(set(_top_k(ranked, k)) & relevant)
    return found / len(relevant)


def precision_at_k(ranked: Sequence[Hashable], relevant: AbstractSet[Hashable], k: int) -> float:
    found = len(set(_top_k(ranked, k)) & relevant)
    return found / k


def hit_at_k(ranked: Sequence[Hashable], relevant: AbstractSet[Hashable], k: int) -> float:
    return 1.0 if set(_top_k(ranked, k)) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[Hashable], relevant: AbstractSet[Hashable]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: Sequence[Hashable], relevant: AbstractSet[Hashable], k: int) -> float:
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(_top_k(ranked, k), start=1)
        if item in relevant
    )


def ndcg_at_k(ranked: Sequence[Hashable], relevant: AbstractSet[Hashable], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(ranked, relevant, k) / idcg


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    has_relevant: bool
    recall: dict[int, float]
    hit: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    latency_ms: float | None = None


@dataclass(frozen=True)
class AggregateScore:
    num_queries: int
    num_adversarial: int
    recall: dict[int, float]
    hit: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    mean_latency_ms: float | None
    p95_latency_ms: float | None


def score_query(
    query_id: str,
    ranked: Sequence[Hashable],
    relevant: AbstractSet[Hashable],
    *,
    ks: Sequence[int],
    latency_ms: float | None = None,
) -> QueryScore:
    return QueryScore(
        query_id=query_id,
        has_relevant=bool(relevant),
        recall={k: recall_at_k(ranked, relevant, k) for k in ks},
        hit={k: hit_at_k(ranked, relevant, k) for k in ks},
        ndcg={k: ndcg_at_k(ranked, relevant, k) for k in ks},
        mrr=reciprocal_rank(ranked, relevant),
        latency_ms=latency_ms,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(pct / 100.0 * len(ordered))) - 1)
    return ordered[max(0, index)]


def aggregate(scores: Sequence[QueryScore], *, ks: Sequence[int]) -> AggregateScore:
    # Quality metrics are averaged only over queries that carry relevant labels;
    # adversarial queries (no expected sources) are counted separately.
    labeled = [score for score in scores if score.has_relevant]
    latencies = [score.latency_ms for score in scores if score.latency_ms is not None]
    return AggregateScore(
        num_queries=len(scores),
        num_adversarial=len(scores) - len(labeled),
        recall={k: _mean([score.recall[k] for score in labeled]) for k in ks},
        hit={k: _mean([score.hit[k] for score in labeled]) for k in ks},
        ndcg={k: _mean([score.ndcg[k] for score in labeled]) for k in ks},
        mrr=_mean([score.mrr for score in labeled]),
        mean_latency_ms=_mean(latencies) if latencies else None,
        p95_latency_ms=_percentile(latencies, 95.0),
    )
