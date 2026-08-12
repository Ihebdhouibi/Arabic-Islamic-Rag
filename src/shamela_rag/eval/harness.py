"""Retrieval evaluation harness (M6-02): score a retriever against the golden set.

``evaluate_retrieval`` runs a ``retrieve`` callable (query -> ranked book ids) over each golden
example, times it, and produces per-query scores plus an aggregate report (recall@k, hit@k,
nDCG@k, MRR, query latency). It is backend-agnostic: pass a lambda that calls the real
``RetrievalService`` in production, or a canned function in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from shamela_rag.eval.dataset import GoldenExample
from shamela_rag.eval.metrics import AggregateScore, QueryScore, aggregate, score_query

DEFAULT_KS = (10, 100)

RetrieveFn = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class EvalReport:
    per_query: tuple[QueryScore, ...]
    aggregate: AggregateScore
    ks: tuple[int, ...]


def evaluate_retrieval(
    dataset: Sequence[GoldenExample],
    retrieve: RetrieveFn,
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> EvalReport:
    if not ks:
        raise ValueError("ks must be non-empty")

    scores: list[QueryScore] = []
    for example in dataset:
        start = time.perf_counter()
        ranked = list(retrieve(example.query))
        latency_ms = (time.perf_counter() - start) * 1000.0
        scores.append(
            score_query(
                example.example_id,
                ranked,
                example.relevant_book_ids,
                ks=ks,
                latency_ms=latency_ms,
            )
        )

    return EvalReport(
        per_query=tuple(scores),
        aggregate=aggregate(scores, ks=ks),
        ks=tuple(ks),
    )


def book_ids_from_hits(hits: Sequence[object]) -> list[int]:
    """Extract ranked book ids from retrieval hits (objects exposing ``book_id``)."""
    ranked: list[int] = []
    for hit in hits:
        book_id = getattr(hit, "book_id", None)
        if isinstance(book_id, int):
            ranked.append(book_id)
    return ranked
