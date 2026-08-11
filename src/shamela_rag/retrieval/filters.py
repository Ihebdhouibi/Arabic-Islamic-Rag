"""Payload filter for retrieval, translated to a Qdrant filter at query time."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models


@dataclass(frozen=True)
class RetrievalFilter:
    book_id: int | None = None
    category_id: int | None = None
    content_role: str | None = None

    def to_qdrant(self) -> models.Filter | None:
        conditions: list[models.Condition] = []
        if self.book_id is not None:
            conditions.append(
                models.FieldCondition(key="book_id", match=models.MatchValue(value=self.book_id))
            )
        if self.category_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="category_id", match=models.MatchValue(value=self.category_id)
                )
            )
        if self.content_role is not None:
            conditions.append(
                models.FieldCondition(
                    key="content_role", match=models.MatchValue(value=self.content_role)
                )
            )
        return models.Filter(must=conditions) if conditions else None
