"""Reciprocal Rank Fusion (RRF) over multiple ranked retrieval lists.

RRF combines the dense and sparse rankings without needing comparable scores: each list contributes
``1 / (k + rank)`` per chunk (rank is 1-based), and contributions are summed. A chunk ranked
moderately in *both* lists can outrank one ranked highly in only one — the property we want from
hybrid retrieval. Ties break by ascending ``chunk_id`` for deterministic output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shamela_rag.retrieval.results import RetrievedChunk

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: int
    score: float
    payload: dict[str, Any]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedChunk]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[int, float] = {}
    payloads: dict[int, dict[str, Any]] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(chunk.chunk_id, chunk.payload)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused = [
        FusedChunk(chunk_id=chunk_id, score=score, payload=payloads[chunk_id])
        for chunk_id, score in ordered
    ]
    return fused[:limit] if limit is not None else fused
