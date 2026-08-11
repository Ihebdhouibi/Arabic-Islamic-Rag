"""Cross-encoder reranking: reorder retrieved candidates by (query, passage) relevance.

Retrievers return candidate ids; the reranker needs the passage *text*, so callers pass
``RerankCandidate`` (id + text) — typically the top ~100 fused hits with text fetched from Postgres.
A multilingual cross-encoder (BGE-reranker-v2-m3) scores each pair and the top ~10 are kept. The
concrete model is behind a flag; ``LexicalOverlapReranker`` is a deterministic offline stand-in for
tests, and the real backend loads lazily (integration tests skip when weights are absent).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from shamela_rag.text.normalization import normalize_for_index

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: int
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedChunk:
    chunk_id: int
    score: float
    text: str
    payload: dict[str, Any]


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return a relevance score for each ``(query, passage)`` pair, in input order."""

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int | None = None
    ) -> list[RerankedChunk]:
        if not candidates:
            return []
        scores = self.score(query, [candidate.text for candidate in candidates])
        if len(scores) != len(candidates):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].chunk_id),
        )
        reranked = [
            RerankedChunk(
                chunk_id=candidate.chunk_id,
                score=float(score),
                text=candidate.text,
                payload=candidate.payload,
            )
            for candidate, score in ordered
        ]
        return reranked[:top_k] if top_k is not None else reranked


class LexicalOverlapReranker(Reranker):
    """Deterministic, offline reranker scoring by shared normalized-token count (test double)."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(normalize_for_index(query).split())
        return [
            float(len(query_tokens & set(normalize_for_index(passage).split())))
            for passage in passages
        ]


def _load_cross_encoder(model_id: str, *, device: str | None, max_length: int) -> Any:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            'CrossEncoderReranker requires optional deps: pip install "shamela-rag[rerank]"'
        ) from exc

    kwargs: dict[str, Any] = {"max_length": max_length}
    if device is not None:
        kwargs["device"] = device
    return CrossEncoder(model_id, **kwargs)


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self._model = _load_cross_encoder(model_id, device=device, max_length=max_length)
        self._batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        predictions = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in predictions]
