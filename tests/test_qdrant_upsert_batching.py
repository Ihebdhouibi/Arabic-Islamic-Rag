"""Upsert batching in the Qdrant store.

A whole book in one request overruns Qdrant's body limit once vectors are 4096-dim, which is how
the hosted embedding backend is configured. These tests pin the batching that prevents it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shamela_rag.vectorstore.qdrant_store import UPSERT_BATCH_SIZE, ChunkPoint, QdrantStore


def _store() -> tuple[QdrantStore, MagicMock]:
    store = QdrantStore.__new__(QdrantStore)  # bypass __init__: no live Qdrant needed
    client = MagicMock()
    store._client = client
    store._collection = "test_collection"
    store._dense_dim = 8
    return store, client


def _points(count: int) -> list[ChunkPoint]:
    return [
        ChunkPoint(
            point_id=i,
            dense=[0.1] * 8,
            sparse_indices=[1],
            sparse_values=[1.0],
            payload={"book_id": 1},
        )
        for i in range(count)
    ]


def test_small_write_is_a_single_request() -> None:
    store, client = _store()
    store.upsert(_points(3))

    assert client.upsert.call_count == 1
    assert len(client.upsert.call_args.kwargs["points"]) == 3


def test_large_write_is_split_into_batches() -> None:
    store, client = _store()
    store.upsert(_points(150), batch_size=64)

    assert client.upsert.call_count == 3
    sizes = [len(call.kwargs["points"]) for call in client.upsert.call_args_list]
    assert sizes == [64, 64, 22]


def test_every_point_is_written_exactly_once() -> None:
    store, client = _store()
    store.upsert(_points(150), batch_size=64)

    written = [p.id for call in client.upsert.call_args_list for p in call.kwargs["points"]]
    assert written == list(range(150))


def test_exact_multiple_does_not_send_an_empty_batch() -> None:
    store, client = _store()
    store.upsert(_points(128), batch_size=64)

    assert client.upsert.call_count == 2
    assert all(call.kwargs["points"] for call in client.upsert.call_args_list)


def test_empty_input_writes_nothing() -> None:
    store, client = _store()
    store.upsert([])

    client.upsert.assert_not_called()


def test_default_batch_size_is_applied() -> None:
    store, client = _store()
    store.upsert(_points(UPSERT_BATCH_SIZE + 1))

    assert client.upsert.call_count == 2


def test_non_positive_batch_size_is_rejected() -> None:
    store, _ = _store()
    with pytest.raises(ValueError, match="batch_size"):
        store.upsert(_points(1), batch_size=0)
