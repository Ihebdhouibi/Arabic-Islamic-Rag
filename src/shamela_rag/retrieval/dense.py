"""Dense retriever: embed the query and run Qdrant dense nearest-neighbor search.

The query is embedded with the same provider used at ingestion; payload filters (book, category,
content role) narrow the search. Returns chunks ranked by cosine similarity.
"""

from __future__ import annotations

from shamela_rag.embeddings.provider import EmbeddingProvider
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.vectorstore.qdrant_store import QdrantStore


class DenseRetriever:
    def __init__(self, *, embedder: EmbeddingProvider, store: QdrantStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(
        self, query: str, *, limit: int = 10, filters: RetrievalFilter | None = None
    ) -> list[RetrievedChunk]:
        vector = self._embedder.embed_query(query)
        query_filter = filters.to_qdrant() if filters is not None else None
        points = self._store.search_dense(vector, limit=limit, query_filter=query_filter)
        return [RetrievedChunk.from_scored_point(point) for point in points]
