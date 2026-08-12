"""Evaluation helpers: structural validation and retrieval metrics."""

from __future__ import annotations

from shamela_rag.eval.dataset import (
    GoldenExample,
    GoldenSource,
    load_golden_dataset,
)
from shamela_rag.eval.harness import EvalReport, book_ids_from_hits, evaluate_retrieval
from shamela_rag.eval.metrics import (
    AggregateScore,
    QueryScore,
    aggregate,
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)

__all__ = [
    "AggregateScore",
    "EvalReport",
    "GoldenExample",
    "GoldenSource",
    "QueryScore",
    "aggregate",
    "book_ids_from_hits",
    "dcg_at_k",
    "evaluate_retrieval",
    "hit_at_k",
    "load_golden_dataset",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_query",
]
