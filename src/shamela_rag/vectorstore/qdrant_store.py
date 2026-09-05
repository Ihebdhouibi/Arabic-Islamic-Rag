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
_ROOT = "root"
ROOT_VECTOR_NAME = _ROOT

# Points per upsert request. At 4096 dims one point is roughly 80KB of JSON, so this keeps a
# request near 5MB, comfortably inside Qdrant's 32MB body limit.
UPSERT_BATCH_SIZE = 64


@dataclass
class ChunkPoint:
    point_id: int
    dense: Sequence[float]
    sparse_indices: Sequence[int] = field(default_factory=list)
    sparse_values: Sequence[float] = field(default_factory=list)
    root_sparse_indices: Sequence[int] = field(default_factory=list)
    root_sparse_values: Sequence[float] = field(default_factory=list)
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
            sparse_vectors_config={
                _SPARSE: models.SparseVectorParams(),
                _ROOT: models.SparseVectorParams(),
            },
        )
        # Index the fields the general module filters on.
        for field_name in ("book_id", "category_id", "content_role"):
            self._client.create_payload_index(
                self._collection, field_name, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)

    def delete_by_filter(self, query_filter: models.Filter) -> None:
        self._client.delete(self._collection, points_selector=query_filter, wait=True)

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def upsert(self, points: Iterable[ChunkPoint], *, batch_size: int = UPSERT_BATCH_SIZE) -> None:
        """Write points in batches.

        Qdrant caps the request body (32MB by default). A 4096-dim vector serializes to roughly
        80KB of JSON, so a whole book in one request overruns that limit and the write is rejected
        outright. Batching keeps each request well inside it regardless of book size or dimension.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batch: list[models.PointStruct] = []
        for point in points:
            batch.append(
                models.PointStruct(
                    id=point.point_id,
                    vector=_point_vectors(point),
                    payload=point.payload,
                )
            )
            if len(batch) >= batch_size:
                self._client.upsert(self._collection, points=batch, wait=True)
                batch = []
        if batch:
            self._client.upsert(self._collection, points=batch, wait=True)

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
        self,
        indices: Sequence[int],
        values: Sequence[float],
        *,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        vector_name: str = _SPARSE,
    ) -> list[models.ScoredPoint]:
        return self._client.query_points(
            self._collection,
            query=models.SparseVector(indices=list(indices), values=list(values)),
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points


def _point_vectors(point: ChunkPoint) -> dict[str, Any]:
    vectors: dict[str, Any] = {
        _DENSE: list(point.dense),
        _SPARSE: models.SparseVector(
            indices=list(point.sparse_indices), values=list(point.sparse_values)
        ),
    }
    if point.root_sparse_indices:
        vectors[_ROOT] = models.SparseVector(
            indices=list(point.root_sparse_indices), values=list(point.root_sparse_values)
        )
    return vectors


def get_qdrant_store() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dense_dim=settings.qdrant_dense_dim,
    )
