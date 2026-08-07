from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import models

from shamela_rag.config import get_settings
from shamela_rag.vectorstore.qdrant_store import ChunkPoint, QdrantStore


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    settings = get_settings()
    instance = QdrantStore(
        url=settings.qdrant_url, collection=f"test_{uuid.uuid4().hex}", dense_dim=4
    )
    try:
        instance.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection error means Qdrant is unavailable
        pytest.skip("Qdrant not reachable")
    instance.ensure_collection()
    try:
        yield instance
    finally:
        instance.delete_collection()


def test_dense_search_returns_nearest(store: QdrantStore) -> None:
    store.upsert(
        [
            ChunkPoint(1, dense=[1.0, 0.0, 0.0, 0.0], payload={"book_id": 1}),
            ChunkPoint(2, dense=[0.0, 1.0, 0.0, 0.0], payload={"book_id": 2}),
        ]
    )
    hits = store.search_dense([0.9, 0.1, 0.0, 0.0], limit=1)
    assert hits[0].id == 1


def test_sparse_search_matches_indices(store: QdrantStore) -> None:
    store.upsert(
        [
            ChunkPoint(1, dense=[1.0, 0.0, 0.0, 0.0], sparse_indices=[10], sparse_values=[2.0]),
            ChunkPoint(2, dense=[0.0, 1.0, 0.0, 0.0], sparse_indices=[20], sparse_values=[3.0]),
        ]
    )
    hits = store.search_sparse([20], [1.0], limit=1)
    assert hits[0].id == 2


def test_payload_filter_restricts_results(store: QdrantStore) -> None:
    store.upsert(
        [
            ChunkPoint(1, dense=[1.0, 0.0, 0.0, 0.0], payload={"book_id": 1}),
            ChunkPoint(2, dense=[0.9, 0.1, 0.0, 0.0], payload={"book_id": 2}),
        ]
    )
    query_filter = models.Filter(
        must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=2))]
    )
    hits = store.search_dense([1.0, 0.0, 0.0, 0.0], limit=5, query_filter=query_filter)
    assert [point.id for point in hits] == [2]
