from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from shamela_rag.config import get_settings
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.vectorstore.qdrant_store import ChunkPoint, QdrantStore

_CORPUS: dict[int, tuple[str, int]] = {
    1: ("قال الشافعي في الرسالة إن القياس أصل من أصول الفقه", 10),
    2: ("قال مالك في الموطأ عن نافع عن ابن عمر", 10),
    3: ("باب الطهارة والوضوء وأحكام المياه", 20),
}


@pytest.fixture
def retriever() -> Iterator[SparseRetriever]:
    encoder = Bm25Encoder().fit(text for text, _ in _CORPUS.values())
    store = QdrantStore(
        url=get_settings().qdrant_url, collection=f"test_sparse_{uuid.uuid4().hex}", dense_dim=2
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection error means Qdrant is unavailable
        pytest.skip("Qdrant not reachable")
    store.ensure_collection()
    points = []
    for chunk_id, (text, book_id) in _CORPUS.items():
        sparse = encoder.encode_document(text)
        points.append(
            ChunkPoint(
                point_id=chunk_id,
                dense=[1.0, 0.0],
                sparse_indices=sparse.indices,
                sparse_values=sparse.values,
                payload={"book_id": book_id, "content_role": "body"},
            )
        )
    store.upsert(points)
    try:
        yield SparseRetriever(encoder=encoder, store=store)
    finally:
        store.delete_collection()


def test_exact_name_query_returns_right_chunk(retriever: SparseRetriever) -> None:
    hits = retriever.search("ما رأي الشافعي", limit=1)
    assert hits[0].chunk_id == 1
    assert hits[0].score > 0.0
    assert hits[0].book_id == 10


def test_out_of_vocabulary_query_returns_empty(retriever: SparseRetriever) -> None:
    assert retriever.search("مصطلح غير موجود") == []


def test_payload_filter_restricts_book(retriever: SparseRetriever) -> None:
    hits = retriever.search("الطهارة", limit=5, filters=RetrievalFilter(book_id=20))
    assert [hit.chunk_id for hit in hits] == [3]
    assert all(hit.book_id == 20 for hit in hits)
