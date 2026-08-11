from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from shamela_rag.config import get_settings
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.vectorstore.qdrant_store import ChunkPoint, QdrantStore

_DIMS = 16
# chunk_id -> (text, book_id)
_DOCS: dict[int, tuple[str, int]] = {
    1: ("قال الشافعي في الرسالة إن القياس أصل", 10),
    2: ("قال مالك في الموطأ عن نافع", 10),
    3: ("باب الطهارة والوضوء وأحكام المياه", 20),
}


@pytest.fixture
def retriever() -> Iterator[DenseRetriever]:
    embedder = InMemoryEmbeddingProvider(dims=_DIMS)
    store = QdrantStore(
        url=get_settings().qdrant_url, collection=f"test_dense_{uuid.uuid4().hex}", dense_dim=_DIMS
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection error means Qdrant is unavailable
        pytest.skip("Qdrant not reachable")
    store.ensure_collection()
    points = [
        ChunkPoint(
            point_id=chunk_id,
            dense=embedder.embed_documents([text])[0],
            payload={"book_id": book_id, "content_role": "body"},
        )
        for chunk_id, (text, book_id) in _DOCS.items()
    ]
    store.upsert(points)
    try:
        yield DenseRetriever(embedder=embedder, store=store)
    finally:
        store.delete_collection()


def test_dense_returns_nearest_chunk(retriever: DenseRetriever) -> None:
    hits = retriever.search(_DOCS[1][0], limit=1)
    assert hits[0].chunk_id == 1
    assert hits[0].score > 0.99  # query equals the document text -> cosine ~1
    assert hits[0].book_id == 10
    assert hits[0].content_role == "body"


def test_dense_returns_ranked_list(retriever: DenseRetriever) -> None:
    hits = retriever.search(_DOCS[3][0], limit=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == 3
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_payload_filter_restricts_book(retriever: DenseRetriever) -> None:
    hits = retriever.search(_DOCS[1][0], limit=5, filters=RetrievalFilter(book_id=20))
    assert [hit.chunk_id for hit in hits] == [3]
    assert all(hit.book_id == 20 for hit in hits)
