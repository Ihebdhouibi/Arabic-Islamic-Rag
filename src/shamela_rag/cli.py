"""Command-line entry point for the shamela-rag package.

Exposes ``ingest`` (M3-08) and ``validate-structure`` (M6-06). Heavy embedding backends are
imported lazily so ``--help`` and argument parsing stay fast and offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from shamela_rag.config import get_settings
from shamela_rag.data.discovery import BookLocation, iter_valid_books

if TYPE_CHECKING:
    from shamela_rag.embeddings.provider import EmbeddingProvider
    from shamela_rag.ingestion.pipeline import IngestionService

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shamela-rag", description="RAG over the Shamela4 classical-Arabic corpus."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser(
        "ingest", help="Chunk and index books into Postgres and Qdrant."
    )
    target = ingest.add_mutually_exclusive_group(required=True)
    target.add_argument("--book", type=int, metavar="ID", help="Ingest a single book by id.")
    target.add_argument(
        "--category", type=int, metavar="ID", help="Ingest every book in a category."
    )
    target.add_argument("--all", action="store_true", help="Ingest the entire corpus.")
    ingest.add_argument("--limit", type=int, default=None, help="Cap the number of books ingested.")
    ingest.add_argument(
        "--dry-run", action="store_true", help="Chunk and count without writing anything."
    )
    ingest.add_argument(
        "--model",
        choices=("bge-m3", "qwen3"),
        default=None,
        help="Dense embedding backend (default: bge-m3).",
    )
    ingest.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Override the corpus root (default: SHAMELA_CORPUS_ROOT / config).",
    )

    validate = subcommands.add_parser(
        "validate-structure",
        help="Run structural chunking checks over a book or a corpus sample.",
    )
    validate.add_argument(
        "--book-dir",
        type=Path,
        default=None,
        help="Validate a single book directory (pages.jsonl + toc + metadata).",
    )
    validate.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Corpus root to sample (default: SHAMELA_CORPUS_ROOT when --book-dir omitted).",
    )
    validate.add_argument(
        "--book",
        type=int,
        metavar="ID",
        default=None,
        help="Restrict corpus validation to one book id.",
    )
    validate.add_argument(
        "--category",
        type=int,
        metavar="ID",
        default=None,
        help="Restrict corpus validation to one category id.",
    )
    validate.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max books to validate under --corpus-root (default: 20; use 0 for no cap).",
    )
    return parser


def _select_locations(args: argparse.Namespace, corpus_root: Path) -> list[BookLocation]:
    locations = list(iter_valid_books(corpus_root))
    if args.book is not None:
        locations = [location for location in locations if location.book_id == args.book]
    elif args.category is not None:
        locations = [location for location in locations if location.category_id == args.category]
    if args.limit is not None:
        locations = locations[: args.limit]
    return locations


def run_ingest(args: argparse.Namespace, service: IngestionService) -> int:
    corpus_root = args.corpus_root or get_settings().corpus_root
    locations = _select_locations(args, corpus_root)
    if not locations:
        logger.error("no matching books found under %s", corpus_root)
        return 1

    total_chunks = 0
    total_points = 0
    for location in locations:
        summary = service.ingest_book(location, dry_run=args.dry_run)
        total_chunks += summary.chunk_count
        total_points += summary.upserted_points
        if summary.skipped:
            status = f"skipped: {summary.skipped_reason}"
        elif summary.dry_run:
            status = "dry-run"
        else:
            status = "ok"
        logger.info(
            "book %s: %d chunks, %d points [%s]",
            summary.book_id,
            summary.chunk_count,
            summary.upserted_points,
            status,
        )
    logger.info(
        "done: %d book(s), %d chunks, %d points", len(locations), total_chunks, total_points
    )
    return 0


def _build_embedder(model: str | None) -> EmbeddingProvider:
    if (model or "bge-m3") == "qwen3":
        from shamela_rag.embeddings.qwen import Qwen3EmbeddingProvider

        return Qwen3EmbeddingProvider()
    from shamela_rag.embeddings.bge_m3 import BgeM3EmbeddingProvider

    return BgeM3EmbeddingProvider()


def _build_service(model: str | None) -> IngestionService:
    from shamela_rag.db.engine import get_sessionmaker
    from shamela_rag.ingestion.pipeline import IngestionService
    from shamela_rag.vectorstore.qdrant_store import QdrantStore

    settings = get_settings()
    embedder = _build_embedder(model)
    store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection, dense_dim=embedder.dims
    )
    return IngestionService(session_factory=get_sessionmaker(), store=store, embedder=embedder)


def run_validate_structure(args: argparse.Namespace) -> int:
    from shamela_rag.eval.structural import format_report, validate_book, validate_corpus

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.book_dir is not None:
        book_report = validate_book(args.book_dir)
        print(format_report(book_report), end="")
        return 0 if book_report.ok else 1

    corpus_root = args.corpus_root or get_settings().corpus_root
    limit = None if args.limit == 0 else args.limit
    corpus_report = validate_corpus(
        corpus_root,
        limit=limit,
        book_id=args.book,
        category_id=args.category,
    )
    print(format_report(corpus_report), end="")
    if not corpus_report.books:
        logger.error("no matching books found under %s", corpus_root)
        return 1
    return 0 if corpus_report.ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args, _build_service(args.model))
    if args.command == "validate-structure":
        return run_validate_structure(args)
    parser.error(f"unknown command: {args.command}")  # required subparser makes this unreachable
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
