"""Qdrant collection schema and access for the general module.

One collection holds a **named dense** vector (cosine) and a **named sparse** vector per chunk, plus
an arbitrary payload (citation/filter fields and parent/child links). Dense retrieval, sparse
retrieval, and payload filtering all run against this single collection.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

from shamela_rag.config import get_settings

_DENSE = "dense"
_SPARSE = "sparse"


@dataclass
class ChunkPoint:
    point_id: int
    dense: Sequence[float]
    sparse_indices: Sequence[int] = field(default_factory=list)
    sparse_values: Sequence[float] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantStore:
    def __init__(self, *, url: str, collection: str, dense_dim: int) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dense_dim = dense_dim

    @property
    def client(self) -> QdrantClient:
        return self._client

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE: models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={_SPARSE: models.SparseVectorParams()},
        )
        # Index the fields the general module filters on.
        for field_name in ("book_id", "category_id", "content_role"):
            self._client.create_payload_index(
                self._collection, field_name, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)

    def upsert(self, points: Iterable[ChunkPoint]) -> None:
        structs = [
            models.PointStruct(
                id=point.point_id,
                vector={
                    _DENSE: list(point.dense),
                    _SPARSE: models.SparseVector(
                        indices=list(point.sparse_indices), values=list(point.sparse_values)
                    ),
                },
                payload=point.payload,
            )
            for point in points
        ]
        self._client.upsert(self._collection, points=structs, wait=True)

    def search_dense(
        self, vector: Sequence[float], *, limit: int = 10, query_filter: models.Filter | None = None
    ) -> list[models.ScoredPoint]:
        return self._client.query_points(
            self._collection,
            query=list(vector),
            using=_DENSE,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

    def search_sparse(
        self, indices: Sequence[int], values: Sequence[float], *, limit: int = 10
    ) -> list[models.ScoredPoint]:
        return self._client.query_points(
            self._collection,
            query=models.SparseVector(indices=list(indices), values=list(values)),
            using=_SPARSE,
            limit=limit,
            with_payload=True,
        ).points


def get_qdrant_store() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dense_dim=settings.qdrant_dense_dim,
    )
