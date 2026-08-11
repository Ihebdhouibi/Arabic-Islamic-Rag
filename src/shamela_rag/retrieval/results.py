"""Common retrieval result type: a chunk hit with its score and payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import models


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    payload: dict[str, Any]

    @classmethod
    def from_scored_point(cls, point: models.ScoredPoint) -> RetrievedChunk:
        return cls(
            chunk_id=int(point.id),
            score=float(point.score),
            payload=dict(point.payload or {}),
        )

    def _int_field(self, key: str) -> int | None:
        value = self.payload.get(key)
        return value if isinstance(value, int) else None

    @property
    def book_id(self) -> int | None:
        return self._int_field("book_id")

    @property
    def category_id(self) -> int | None:
        return self._int_field("category_id")

    @property
    def section_id(self) -> int | None:
        return self._int_field("section_id")

    @property
    def page_id(self) -> int | None:
        return self._int_field("page_id")

    @property
    def content_role(self) -> str | None:
        value = self.payload.get("content_role")
        return value if isinstance(value, str) else None
