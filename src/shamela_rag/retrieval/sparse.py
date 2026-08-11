"""Sparse retriever: encode the query with the surface BM25 arm and run Qdrant sparse search.

The query is tokenized and weighted by the same fitted ``Bm25Encoder`` used at ingestion, so exact
surface terms (a scholar's name, a book title) match precisely. Out-of-vocabulary queries return no
hits rather than an error.
"""

from __future__ import annotations

from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.vectorstore.qdrant_store import QdrantStore


class SparseRetriever:
    def __init__(self, *, encoder: Bm25Encoder, store: QdrantStore) -> None:
        self._encoder = encoder
        self._store = store

    def search(
        self, query: str, *, limit: int = 10, filters: RetrievalFilter | None = None
    ) -> list[RetrievedChunk]:
        sparse = self._encoder.encode_query(query)
        if not sparse.indices:
            return []
        query_filter = filters.to_qdrant() if filters is not None else None
        points = self._store.search_sparse(
            sparse.indices, sparse.values, limit=limit, query_filter=query_filter
        )
        return [RetrievedChunk.from_scored_point(point) for point in points]
