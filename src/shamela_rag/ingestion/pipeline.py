"""Per-book ingestion: chunk -> dense-embed -> sparse-encode -> upsert Qdrant + write Postgres.

Idempotent and resumable per book. A book's Postgres rows (sections, chunks) and Qdrant points are
**replaced** on every run (delete-by-book, then insert), so re-ingesting never duplicates. The
verbatim ``source_text`` is written to Postgres alongside provenance/metadata; dense + surface-BM25
sparse vectors go to Qdrant with a payload that links each point back to its Postgres chunk. A
``dry_run`` chunks and counts without writing anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import models
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from shamela_rag.chunking.orchestrator import BookChunk, chunk_book
from shamela_rag.chunking.sections import Section as ChunkSection
from shamela_rag.chunking.sections import build_sections
from shamela_rag.data.discovery import BookLocation, iter_valid_books
from shamela_rag.data.models import Book as BookMeta
from shamela_rag.data.models import load_book, load_toc
from shamela_rag.db.models import Book, Chunk, Section
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import EmbeddingProvider
from shamela_rag.embeddings.root_field import RootExpansionEncoder
from shamela_rag.vectorstore.qdrant_store import ChunkPoint, QdrantStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BookIngestSummary:
    book_id: int
    section_count: int
    chunk_count: int
    upserted_points: int
    dry_run: bool
    skipped_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


def dense_input(chunk: BookChunk) -> str:
    """Text embedded for dense retrieval: context header prepended to the normalized child."""
    if chunk.context_header:
        return f"{chunk.context_header}\n{chunk.retrieval_text}"
    return chunk.retrieval_text


class IngestionService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        store: QdrantStore,
        embedder: EmbeddingProvider,
        sparse_encoder: Bm25Encoder | None = None,
        root_encoder: RootExpansionEncoder | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._store = store
        self._embedder = embedder
        # A shared, pre-fitted encoder keeps sparse vectors comparable across books and matches
        # the query-time encoder; when omitted, each book is fitted on its own chunks.
        self._sparse_encoder = sparse_encoder
        self._root_encoder = root_encoder

    @property
    def root_encoder(self) -> RootExpansionEncoder | None:
        return self._root_encoder

    def ingest_book(self, location: BookLocation, *, dry_run: bool = False) -> BookIngestSummary:
        if not location.has_all_files:
            logger.warning("skipping book %s: missing required files", location.book_id)
            return BookIngestSummary(
                location.book_id, 0, 0, 0, dry_run=dry_run, skipped_reason="missing_files"
            )

        book_meta = load_book(location.book_dir)
        sections = build_sections(list(load_toc(location.book_dir)))
        chunks = chunk_book(location.book_dir).chunks

        if dry_run:
            logger.info(
                "dry-run book %s: %d sections, %d chunks",
                location.book_id,
                len(sections),
                len(chunks),
            )
            return BookIngestSummary(location.book_id, len(sections), len(chunks), 0, dry_run=True)

        if self._sparse_encoder is not None:
            sparse_encoder = self._sparse_encoder
        else:
            sparse_encoder = Bm25Encoder()
            if chunks:
                sparse_encoder.fit(chunk.retrieval_text for chunk in chunks)

        root_encoder = self._root_encoder
        if root_encoder is not None and not root_encoder.is_fitted and chunks:
            root_encoder.fit(chunk.retrieval_text for chunk in chunks)

        self._store.ensure_collection()
        points = self._write_postgres_and_build_points(
            book_meta, location, sections, chunks, sparse_encoder, root_encoder
        )
        self._replace_qdrant_points(location.book_id, points)

        logger.info(
            "ingested book %s: %d sections, %d chunks, %d points",
            location.book_id,
            len(sections),
            len(chunks),
            len(points),
        )
        return BookIngestSummary(
            location.book_id, len(sections), len(chunks), len(points), dry_run=False
        )

    def ingest_corpus(
        self,
        corpus_root: Path,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        skip_existing: bool = False,
    ) -> list[BookIngestSummary]:
        summaries: list[BookIngestSummary] = []
        for location in iter_valid_books(corpus_root):
            if limit is not None and len(summaries) >= limit:
                break
            if skip_existing and not dry_run and self._already_ingested(location.book_id):
                logger.info("skipping already-ingested book %s", location.book_id)
                summaries.append(
                    BookIngestSummary(
                        location.book_id, 0, 0, 0, dry_run=False, skipped_reason="already_ingested"
                    )
                )
                continue
            try:
                summaries.append(self.ingest_book(location, dry_run=dry_run))
            except Exception:
                logger.exception("failed to ingest book %s", location.book_id)
                summaries.append(
                    BookIngestSummary(
                        location.book_id, 0, 0, 0, dry_run=dry_run, skipped_reason="error"
                    )
                )
        return summaries

    def _already_ingested(self, book_id: int) -> bool:
        with self._session_factory() as session:
            return session.query(Chunk.id).filter(Chunk.book_id == book_id).first() is not None

    def _write_postgres_and_build_points(
        self,
        book_meta: BookMeta,
        location: BookLocation,
        sections: list[ChunkSection],
        chunks: list[BookChunk],
        sparse_encoder: Bm25Encoder,
        root_encoder: RootExpansionEncoder | None = None,
    ) -> list[ChunkPoint]:
        dense_vectors = self._embedder.embed_documents([dense_input(c) for c in chunks])
        with self._session_factory() as session, session.begin():
            self._upsert_book(session, book_meta, location)
            session.execute(delete(Chunk).where(Chunk.book_id == location.book_id))
            session.execute(delete(Section).where(Section.book_id == location.book_id))
            section_id_by_trail = self._insert_sections(session, location.book_id, sections)
            chunk_rows = self._insert_chunks(
                session, location.book_id, chunks, section_id_by_trail, book_meta
            )
            session.flush()
            points = self._build_points(
                location, chunks, chunk_rows, dense_vectors, sparse_encoder, root_encoder
            )
        return points

    @staticmethod
    def _upsert_book(session: Session, book_meta: BookMeta, location: BookLocation) -> None:
        row = session.get(Book, book_meta.book_id)
        if row is None:
            row = Book(book_id=book_meta.book_id)
            session.add(row)
        row.title_ar = book_meta.title_ar
        row.author_name_ar = book_meta.main_author_name_ar
        row.author_death_hijri = book_meta.main_author_death_hijri
        row.category_id = (
            book_meta.category_id if book_meta.category_id is not None else location.category_id
        )
        row.book_type_label = book_meta.book_type_label

    @staticmethod
    def _insert_sections(
        session: Session, book_id: int, sections: list[ChunkSection]
    ) -> dict[tuple[str, ...], int]:
        id_by_trail: dict[tuple[str, ...], int] = {}
        for section in sorted(sections, key=lambda s: s.depth):
            parent_id = id_by_trail.get(section.trail[:-1]) if section.depth > 0 else None
            row = Section(
                book_id=book_id,
                parent_id=parent_id,
                shamela_title_id=section.shamela_title_id,
                title_text=section.title_text,
                title_trail=" > ".join(section.trail),
                depth=section.depth,
                path_source=section.path_source.value,
                start_page_id=section.start_page_id,
                end_page_id=section.end_page_id,
            )
            session.add(row)
            session.flush()
            id_by_trail[section.trail] = row.id
        return id_by_trail

    @staticmethod
    def _resolve_section_id(
        chunk: BookChunk, id_by_trail: dict[tuple[str, ...], int], book_meta: BookMeta
    ) -> int | None:
        if chunk.trail in id_by_trail:
            return id_by_trail[chunk.trail]
        # A chunk trail may carry the book title as its root; sections do not.
        if (
            chunk.trail
            and book_meta.title_ar
            and chunk.trail[0] == book_meta.title_ar
            and chunk.trail[1:] in id_by_trail
        ):
            return id_by_trail[chunk.trail[1:]]
        return None

    @classmethod
    def _insert_chunks(
        cls,
        session: Session,
        book_id: int,
        chunks: list[BookChunk],
        id_by_trail: dict[tuple[str, ...], int],
        book_meta: BookMeta,
    ) -> list[Chunk]:
        rows: list[Chunk] = []
        for chunk in chunks:
            row = Chunk(
                book_id=book_id,
                section_id=cls._resolve_section_id(chunk, id_by_trail, book_meta),
                content_role=chunk.content_role.value,
                source_text=chunk.source_text,
                retrieval_text=chunk.retrieval_text,
                context_header=chunk.context_header,
                start_page_id=chunk.page_id,
                end_page_id=chunk.page_id,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                token_count=chunk.token_count,
            )
            session.add(row)
            rows.append(row)
        session.flush()
        return rows

    @staticmethod
    def _build_points(
        location: BookLocation,
        chunks: list[BookChunk],
        chunk_rows: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_encoder: Bm25Encoder,
        root_encoder: RootExpansionEncoder | None = None,
    ) -> list[ChunkPoint]:
        points: list[ChunkPoint] = []
        for chunk, row, dense in zip(chunks, chunk_rows, dense_vectors, strict=True):
            sparse = sparse_encoder.encode_document(chunk.retrieval_text)
            root = (
                root_encoder.encode_document(chunk.retrieval_text)
                if root_encoder is not None
                else None
            )
            points.append(
                ChunkPoint(
                    point_id=row.id,
                    dense=dense,
                    sparse_indices=sparse.indices,
                    sparse_values=sparse.values,
                    root_sparse_indices=root.indices if root is not None else (),
                    root_sparse_values=root.values if root is not None else (),
                    payload={
                        "chunk_id": row.id,
                        "book_id": location.book_id,
                        "category_id": location.category_id,
                        "section_id": row.section_id,
                        "content_role": chunk.content_role.value,
                        "page_id": chunk.page_id,
                    },
                )
            )
        return points

    def _replace_qdrant_points(self, book_id: int, points: list[ChunkPoint]) -> None:
        self._store.delete_by_filter(
            models.Filter(
                must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))]
            )
        )
        if points:
            self._store.upsert(points)
