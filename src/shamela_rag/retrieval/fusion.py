"""Reciprocal Rank Fusion (RRF) over multiple ranked retrieval lists.

RRF combines the dense and sparse rankings without needing comparable scores: each list contributes
``1 / (k + rank)`` per chunk (rank is 1-based), and contributions are summed. A chunk ranked
moderately in *both* lists can outrank one ranked highly in only one — the property we want from
hybrid retrieval. Ties break by ascending ``chunk_id`` for deterministic output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from shamela_rag.retrieval.results import RetrievedChunk

DEFAULT_RRF_K = 60

ARM_DENSE = "dense"
ARM_BM25 = "bm25"
ARM_ROOT = "root"
ARM_RERANK = "rerank"

_SOURCE_ORDER = {ARM_DENSE: 0, ARM_BM25: 1, ARM_ROOT: 2, ARM_RERANK: 3}
RETRIEVAL_SOURCES_KEY = "retrieval_sources"


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: int
    score: float
    payload: dict[str, Any]


def normalize_retrieval_sources(sources: Iterable[str]) -> list[str]:
    return sorted(set(sources), key=lambda name: (_SOURCE_ORDER.get(name, 99), name))


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
    arm_names: Sequence[str] | None = None,
) -> list[FusedChunk]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if arm_names is not None and len(arm_names) != len(result_lists):
        raise ValueError(
            f"arm_names length {len(arm_names)} must match result_lists length {len(result_lists)}"
        )

    scores: dict[int, float] = {}
    payloads: dict[int, dict[str, Any]] = {}
    arms: dict[int, set[str]] = {}
    for list_index, results in enumerate(result_lists):
        arm = arm_names[list_index] if arm_names is not None else None
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(chunk.chunk_id, chunk.payload)
            if arm is not None:
                arms.setdefault(chunk.chunk_id, set()).add(arm)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused: list[FusedChunk] = []
    for chunk_id, score in ordered:
        payload = dict(payloads[chunk_id])
        if arm_names is not None:
            payload[RETRIEVAL_SOURCES_KEY] = normalize_retrieval_sources(arms.get(chunk_id, ()))
        fused.append(FusedChunk(chunk_id=chunk_id, score=score, payload=payload))
    return fused[:limit] if limit is not None else fused
