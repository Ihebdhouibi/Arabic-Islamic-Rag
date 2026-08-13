"""Fit one surface-BM25 encoder over many books (M3-13).

Chunks each book (offline; no DB/vector services) and fits a single ``Bm25Encoder`` on all
``retrieval_text``, so every book shares one sparse term space that matches the query-time encoder.
Books that fail to chunk are skipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from shamela_rag.chunking.navigation import DEFAULT_MIN_CONTENT_TOKENS
from shamela_rag.chunking.orchestrator import chunk_book
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.logging_config import get_logger

logger = get_logger(__name__)


def _iter_retrieval_texts(book_dirs: Iterable[Path], *, min_content_tokens: int) -> Iterator[str]:
    for book_dir in book_dirs:
        try:
            result = chunk_book(book_dir, min_content_tokens=min_content_tokens)
        except Exception:
            logger.exception("failed to chunk %s for BM25 fit", book_dir)
            continue
        for chunk in result.chunks:
            yield chunk.retrieval_text


def fit_corpus_bm25(
    book_dirs: Iterable[Path], *, min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS
) -> Bm25Encoder:
    """Fit one BM25 encoder across all chunks of ``book_dirs`` (raises on an empty corpus)."""
    return Bm25Encoder().fit(
        _iter_retrieval_texts(book_dirs, min_content_tokens=min_content_tokens)
    )
