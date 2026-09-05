"""Command-line entry point for the shamela-rag package.

Exposes ``ingest`` (M3-08), ``validate-structure`` (M6-06), ``audit-structure`` (#149),
``build-bm25``, ``ask``,
``compare-qwen-quant`` (issue #135), and ``compare-dense-models`` (M6-03). Heavy embedding
backends are imported lazily so ``--help`` and argument parsing stay fast and offline.
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

    audit = subcommands.add_parser(
        "audit-structure",
        help="Per-category structural boundary audit over a stratified corpus sample.",
    )
    audit.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Corpus root to sample (default: SHAMELA_CORPUS_ROOT).",
    )
    audit.add_argument(
        "--books-per-category",
        type=int,
        default=1,
        help="Books to sample per category (default: 1).",
    )
    audit.add_argument(
        "--book",
        type=int,
        metavar="ID",
        default=None,
        help="Restrict the stratified sample to one book id.",
    )
    audit.add_argument(
        "--category",
        type=int,
        metavar="ID",
        default=None,
        help="Restrict the stratified sample to one category id.",
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown report to this path (default: stdout only).",
    )

    bench = subcommands.add_parser(
        "benchmark",
        help="Sizing and latency benchmark over a stratified ingested sample (#148).",
    )
    bench.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Corpus root to sample (default: SHAMELA_CORPUS_ROOT).",
    )
    bench.add_argument(
        "--books-per-category",
        type=int,
        default=1,
        help="Books to ingest per category for the sample (default: 1).",
    )
    bench.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Measure whatever is already in Postgres/Qdrant instead of ingesting first.",
    )
    bench.add_argument(
        "--skip-latency",
        action="store_true",
        help="Only measure sizing (the half that does not need a working retrieval stack).",
    )
    bench.add_argument(
        "--budget-ms",
        type=int,
        default=8000,
        help="Latency budget the p95 is judged against (default: 8000).",
    )
    bench.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Newline-separated queries for the latency run (default: a built-in Arabic set).",
    )
    bench.add_argument(
        "--qdrant-container",
        default="shamela-qdrant",
        help="Container name used to measure Qdrant disk usage (default: shamela-qdrant).",
    )
    bench.add_argument(
        "--model",
        choices=("bge-m3", "qwen3"),
        default=None,
        help="Dense embedding backend used for ingest and query (default: bge-m3).",
    )
    bench.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown report to this path (default: stdout only).",
    )

    build_bm25 = subcommands.add_parser(
        "build-bm25", help="Fit a corpus-wide surface-BM25 encoder and persist it."
    )
    bm25_target = build_bm25.add_mutually_exclusive_group(required=True)
    bm25_target.add_argument("--book", type=int, metavar="ID", help="Fit over a single book id.")
    bm25_target.add_argument(
        "--category", type=int, metavar="ID", help="Fit over every book in a category."
    )
    bm25_target.add_argument("--all", action="store_true", help="Fit over the entire corpus.")
    build_bm25.add_argument("--limit", type=int, default=None, help="Cap the number of books.")
    build_bm25.add_argument(
        "--corpus-root", type=Path, default=None, help="Override the corpus root."
    )
    build_bm25.add_argument(
        "--output",
        type=Path,
        default=None,
        help="State file path (default: config bm25_state_path).",
    )

    ask = subcommands.add_parser("ask", help="Answer a question against the ingested corpus.")
    ask.add_argument("question", help="The question (Arabic or English).")
    ask.add_argument("--k", type=int, default=None, help="Number of passages to cite.")
    ask.add_argument("--book", type=int, metavar="ID", default=None, help="Restrict to one book.")
    ask.add_argument(
        "--category", type=int, metavar="ID", default=None, help="Restrict to one category."
    )
    ask.add_argument(
        "--model", choices=("bge-m3", "qwen3"), default=None, help="Dense embedding backend."
    )
    ask.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder (offline lexical reranker).",
    )
    ask.add_argument("--json", action="store_true", help="Emit the answer as JSON.")

    quant = subcommands.add_parser(
        "compare-qwen-quant",
        help="Compare Qwen3 fp16 vs int8/int4/GGUF footprint and embedding quality (#135).",
    )
    quant.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for comparison_table.md and metrics.json.",
    )
    quant.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device for the fp16 baseline (e.g. cuda, cpu).",
    )
    quant.add_argument(
        "--skip-fp16",
        action="store_true",
        help="Skip fp16 baseline (required on ~16GB RAM / no-GPU hosts that OOM on load).",
    )
    quant.add_argument(
        "--no-int8",
        action="store_true",
        help="Skip bitsandbytes int8 (CUDA-oriented; fails on CPU-only machines).",
    )
    quant.add_argument(
        "--no-int4",
        action="store_true",
        help="Skip the int4 (stretch) arm.",
    )
    quant.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Local Qwen3-Embedding GGUF path (llama.cpp embedding mode).",
    )
    quant.add_argument(
        "--gguf-baseline",
        type=Path,
        default=None,
        help="Higher-bit GGUF used as cosine baseline when fp16 is skipped (e.g. Q8_0).",
    )
    quant.add_argument(
        "--download-gguf",
        action="store_true",
        help=(
            "Download official Qwen3-Embedding-8B Q4_K_M.gguf into HF_HOME "
            "(or --gguf-dir) then run that arm."
        ),
    )
    quant.add_argument(
        "--download-gguf-q8",
        action="store_true",
        help="Download official Q8_0.gguf as --gguf-baseline (CPU proxy when fp16 OOMs).",
    )
    quant.add_argument(
        "--gguf-dir",
        type=Path,
        default=None,
        help="Directory for GGUF downloads (default: HF_HOME, else Hugging Face cache).",
    )
    quant.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Shamela corpus root; chunk golden-set books for retrieval (with --golden).",
    )
    quant.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Rebuild eval_chunks.jsonl even when a cache exists under --output-dir.",
    )
    quant.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Optional chunk JSONL for dense retrieval metrics (text + book_id).",
    )
    quant.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Golden JSONL for retrieval metrics (required with --corpus-root or --chunks).",
    )
    quant.add_argument(
        "--max-chunks",
        type=int,
        default=512,
        help="Cap eval chunks after load/build (default: 512; round-robin per book).",
    )
    quant.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help="Dense candidate limit for retrieval arm (default: 100).",
    )
    quant.add_argument(
        "--gguf-n-ctx",
        type=int,
        default=512,
        help="GGUF context length / truncate budget (default: 512; lower = faster CPU embeds).",
    )

    dense = subcommands.add_parser(
        "compare-dense-models",
        help="M6-03: compare Qwen3 vs BGE-M3 retrieval on the golden set.",
    )
    dense.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for dense_only_table.md, metrics, and embedding caches.",
    )
    dense.add_argument(
        "--golden",
        type=Path,
        default=Path("docs/technical_docs/general_qa_golden_staging.jsonl"),
        help="Golden JSONL (default: staging set).",
    )
    dense.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Eval chunk JSONL (text + book_id).",
    )
    dense.add_argument(
        "--max-chunks",
        type=int,
        default=2048,
        help="Round-robin chunk subsample cap (default: 2048).",
    )
    dense.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help="Dense candidate book limit (default: 100).",
    )
    dense.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size (default: 16).",
    )
    dense.add_argument(
        "--force-reembed",
        action="store_true",
        help="Ignore embedding caches under --output-dir.",
    )
    dense.add_argument(
        "--stage",
        choices=("dense-only", "hybrid-bm25", "bge-sparse", "both"),
        default="dense-only",
        help="Stage: dense-only, hybrid-bm25, bge-sparse, or both (stages 1+2).",
    )
    dense.add_argument(
        "--fusion-pool",
        type=int,
        default=200,
        help="Per-list pool size before RRF in hybrid stage (default: 200).",
    )
    dense.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k for hybrid fusion (default: 60).",
    )
    dense.add_argument(
        "--sparse-batch-size",
        type=int,
        default=8,
        help="BGE-M3 learned-sparse batch size for bge-sparse stage (default: 8).",
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
    settings = get_settings()
    if (
        not args.dry_run
        and settings.root_expansion_enabled
        and service.root_encoder is not None
        and service.root_encoder.is_fitted
    ):
        service.root_encoder.save(settings.root_expansion_state_path)
        logger.info("wrote root expansion state to %s", settings.root_expansion_state_path)
    return 0


def _build_embedder(model: str | None) -> EmbeddingProvider:
    """Delegate to ``factory.build_embedder`` so ingestion respects ``SHAMELA_EMBEDDING_BACKEND``
    the same way the query-time service does — otherwise ``ingest``/``build-bm25`` would silently
    load local weights even with the OpenRouter backend configured."""
    from shamela_rag.factory import build_embedder

    return build_embedder(model)


def _build_service(model: str | None) -> IngestionService:
    from shamela_rag.db.engine import get_sessionmaker
    from shamela_rag.embeddings.bm25 import Bm25Encoder
    from shamela_rag.ingestion.pipeline import IngestionService
    from shamela_rag.vectorstore.qdrant_store import QdrantStore

    settings = get_settings()
    embedder = _build_embedder(model)
    store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection, dense_dim=embedder.dims
    )
    # Reuse a persisted corpus-wide BM25 encoder when present so sparse vectors stay comparable.
    state_path = settings.bm25_state_path
    encoder = Bm25Encoder.load(state_path) if state_path.exists() else None
    if encoder is not None:
        logger.info("using persisted BM25 encoder from %s", state_path)
    root_encoder = None
    if settings.root_expansion_enabled:
        from shamela_rag.data.root_dictionary import load_root_dictionary
        from shamela_rag.embeddings.root_field import RootExpansionEncoder

        dictionary = load_root_dictionary(settings.resolved_root_dictionary_path)
        root_state = settings.root_expansion_state_path
        if root_state.exists():
            root_encoder = RootExpansionEncoder.load(root_state, dictionary)
            logger.info("using persisted root expansion encoder from %s", root_state)
        else:
            root_encoder = RootExpansionEncoder(dictionary, weight=settings.root_expansion_weight)
            logger.info("root expansion enabled from %s", settings.resolved_root_dictionary_path)
    return IngestionService(
        session_factory=get_sessionmaker(),
        store=store,
        embedder=embedder,
        sparse_encoder=encoder,
        root_encoder=root_encoder,
    )


_DEFAULT_BENCHMARK_QUERIES = (
    "ما حكم الصلاة في السفر؟",
    "ما هي شروط صحة البيع؟",
    "من هو الإمام الشافعي؟",
    "ما معنى الإجماع عند الأصوليين؟",
    "ما حكم صيام يوم عرفة؟",
    "ما الفرق بين الحديث الصحيح والحسن؟",
    "ما هي أركان الإيمان؟",
    "ما حكم الزكاة في عروض التجارة؟",
    "كيف تقسم الفرائض بين الورثة؟",
    "ما هي مقاصد الشريعة؟",
    "ما حكم الوضوء من لحم الإبل؟",
    "ما هو تعريف القياس في أصول الفقه؟",
)


def run_benchmark(args: argparse.Namespace) -> int:
    """Ingest a stratified sample, measure both stores, and time the retrieval pipeline."""
    from shamela_rag.db.engine import get_engine
    from shamela_rag.eval.benchmark import (
        BenchmarkReport,
        docker_directory_size,
        format_benchmark_report,
        measure_latency,
        measure_postgres,
        measure_qdrant,
    )
    from shamela_rag.eval.structural import stratified_book_locations

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def status(message: str) -> None:
        print(f"[benchmark] {message}", flush=True)
        logger.info("%s", message)

    settings = get_settings()
    corpus_root = args.corpus_root or settings.corpus_root
    notes: list[str] = []
    categories: tuple[int, ...] = ()

    if args.skip_ingest:
        status("skipping ingest, measuring what is already stored")
        notes.append("Measured an already-ingested database; this run did not ingest.")
    else:
        status(f"selecting {args.books_per_category} book(s) per category under {corpus_root}")
        locations = stratified_book_locations(
            corpus_root, books_per_category=args.books_per_category
        )
        if not locations:
            logger.error("no matching books found under %s", corpus_root)
            return 1
        categories = tuple(
            sorted({loc.category_id for loc in locations if loc.category_id is not None})
        )
        status(f"ingesting {len(locations)} book(s) across {len(categories)} categories")

        service = _build_service(args.model)
        for index, location in enumerate(locations, start=1):
            summary = service.ingest_book(location)
            status(
                f"[{index}/{len(locations)}] book {summary.book_id}: "
                f"{summary.chunk_count} chunks, {summary.upserted_points} points"
            )
        notes.append(
            f"Sample ingested with {args.books_per_category} book(s) per category "
            f"({len(locations)} books)."
        )

    status("measuring Postgres")
    postgres = measure_postgres(get_engine())

    status("measuring Qdrant")
    from shamela_rag.vectorstore.qdrant_store import QdrantStore

    store = QdrantStore()
    disk = docker_directory_size(args.qdrant_container)
    if disk is None:
        notes.append(
            "Qdrant disk size unavailable (container not reachable); point count is still measured."
        )
    qdrant = measure_qdrant(store.client, settings.qdrant_collection, storage_path=None)
    qdrant = type(qdrant)(
        collection=qdrant.collection,
        points=qdrant.points,
        disk_bytes=disk,
        vector_dim=qdrant.vector_dim,
    )

    latency = None
    if not args.skip_latency:
        if args.queries_file is not None:
            queries = [
                line.strip()
                for line in args.queries_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            queries = list(_DEFAULT_BENCHMARK_QUERIES)
        status(f"timing {len(queries)} queries against the retrieval pipeline")

        from shamela_rag.factory import build_general_qa_service

        qa = build_general_qa_service(model=args.model or "bge-m3")
        latency = measure_latency(qa._retrieval, queries, budget_ms=args.budget_ms)
        status(f"p95 total: {latency.total.p95:.0f} ms (budget {args.budget_ms} ms)")

    report = BenchmarkReport(
        postgres=postgres,
        qdrant=qdrant,
        latency=latency,
        sample_categories=categories,
        notes=tuple(notes),
    )
    markdown = format_benchmark_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        status(f"wrote report to {args.output}")
    print(markdown, end="")
    return 0


def run_build_bm25(args: argparse.Namespace) -> int:
    from shamela_rag.ingestion.bm25_fit import fit_corpus_bm25

    settings = get_settings()
    corpus_root = args.corpus_root or settings.corpus_root
    locations = _select_locations(args, corpus_root)
    if not locations:
        logger.error("no matching books found under %s", corpus_root)
        return 1

    try:
        encoder = fit_corpus_bm25([location.book_dir for location in locations])
    except ValueError:
        logger.error("no chunkable content in the selected books; nothing to fit")
        return 1

    output = args.output or settings.bm25_state_path
    encoder.save(output)
    logger.info(
        "fitted BM25 over %d book(s), %d terms -> %s",
        len(locations),
        encoder.vocabulary_size,
        output,
    )
    return 0


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


def run_audit_structure(args: argparse.Namespace) -> int:
    from shamela_rag.eval.structural import (
        format_category_audit_report,
        format_report,
        validate_category_audit,
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def status(message: str) -> None:
        print(f"[audit-structure] {message}", flush=True)
        logger.info("%s", message)

    corpus_root = args.corpus_root or get_settings().corpus_root
    corpus_report, audit_report = validate_category_audit(
        corpus_root,
        books_per_category=args.books_per_category,
        book_id=args.book,
        category_id=args.category,
        progress=status,
    )
    if not corpus_report.books:
        logger.error("no matching books found under %s", corpus_root)
        return 1

    markdown = format_category_audit_report(audit_report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        status(f"wrote report to {args.output}")
    print(markdown, end="")
    if not corpus_report.ok:
        print(format_report(corpus_report), end="")
        return 1
    return 0


def run_compare_qwen_quant(args: argparse.Namespace) -> int:
    import os

    from shamela_rag.embeddings.qwen import download_qwen_gguf
    from shamela_rag.eval.qwen_quant import (
        default_variant_specs,
        format_quant_table,
        run_qwen_quant_comparison,
        write_quant_artifacts,
    )

    def status(message: str) -> None:
        print(f"[compare-qwen-quant] {message}", flush=True)
        logger.info("%s", message)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.chunks is not None and args.golden is None:
        logger.error("--golden is required when --chunks is set")
        return 1
    if args.corpus_root is not None and args.golden is None:
        logger.error("--golden is required when --corpus-root is set")
        return 1

    gguf_path = args.gguf
    gguf_baseline = args.gguf_baseline
    hf_home = os.environ.get("HF_HOME")
    gguf_dir: Path | None = args.gguf_dir or (Path(hf_home) if hf_home else None)

    try:
        if args.download_gguf:
            status(f"downloading official Q4_K_M GGUF into {gguf_dir or 'Hugging Face cache'}")
            gguf_path = download_qwen_gguf(local_dir=gguf_dir)
            status(f"GGUF ready at {gguf_path}")
        if args.download_gguf_q8:
            from shamela_rag.embeddings.qwen import QWEN3_EMBEDDING_GGUF_Q8_0

            status(f"downloading official Q8_0 GGUF into {gguf_dir or 'Hugging Face cache'}")
            gguf_baseline = download_qwen_gguf(
                filename=QWEN3_EMBEDDING_GGUF_Q8_0, local_dir=gguf_dir
            )
            status(f"GGUF Q8 baseline ready at {gguf_baseline}")
    except Exception as exc:  # noqa: BLE001 - surface download/auth failures cleanly
        logger.error("GGUF download failed: %s", exc)
        return 1

    if not args.skip_fp16 and not args.no_int8 and gguf_path is None:
        status(
            "tip: on 16GB RAM / no GPU use "
            "--skip-fp16 --no-int8 --no-int4 --gguf <q4> --download-gguf-q8"
        )

    try:
        specs = default_variant_specs(
            include_fp16=not args.skip_fp16,
            include_int8=not args.no_int8,
            include_int4=not args.no_int4,
            gguf_path=gguf_path,
            gguf_baseline_path=gguf_baseline,
            device=args.device,
            gguf_n_ctx=args.gguf_n_ctx,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    status(f"variants: {[s.name for s in specs]}")
    report = run_qwen_quant_comparison(
        specs,
        chunks_path=args.chunks,
        corpus_root=args.corpus_root,
        golden_path=args.golden,
        chunk_cache_path=args.output_dir / "eval_chunks.jsonl",
        force_rechunk=args.force_rechunk,
        candidate_limit=args.candidate_limit,
        max_chunks=args.max_chunks,
        progress=status,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_quant_artifacts(args.output_dir, report)
    print(format_quant_table(report), end="")
    status(f"done — see {args.output_dir / 'comparison_table.md'}")
    return 0


def run_compare_dense_models(args: argparse.Namespace) -> int:
    """Run M6-03 dense-model comparison stages."""
    from shamela_rag.embeddings.bge_m3 import BgeM3EmbeddingProvider
    from shamela_rag.eval.comparison import format_comparison
    from shamela_rag.eval.model_ab import (
        run_bge_sparse_ablation,
        run_dense_only_comparison,
        run_hybrid_bm25_comparison,
    )
    from shamela_rag.factory import build_embedder

    def status(message: str) -> None:
        print(f"[compare-dense-models] {message}", flush=True)
        logger.info("%s", message)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.chunks.is_file():
        logger.error("chunks file not found: %s", args.chunks)
        return 1
    if not args.golden.is_file():
        logger.error("golden file not found: %s", args.golden)
        return 1

    settings = get_settings()
    status(f"embedding_backend={settings.embedding_backend} stage={args.stage}")
    try:
        # bge-sparse only ever uses the bge-m3 dense provider; skip building qwen3 for it so a
        # broken/unavailable qwen3 backend doesn't block a pure bge-sparse run.
        models = {"bge-m3": build_embedder("bge-m3")}
        if args.stage != "bge-sparse":
            models["qwen3"] = build_embedder("qwen3")
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to build embedders: %s", exc)
        return 1

    last_report = None
    try:
        if args.stage in ("dense-only", "both"):
            last_report = run_dense_only_comparison(
                models=models,
                golden_path=args.golden,
                chunks_path=args.chunks,
                output_dir=args.output_dir,
                max_chunks=args.max_chunks,
                candidate_limit=args.candidate_limit,
                batch_size=args.batch_size,
                force_reembed=args.force_reembed,
                progress=status,
            )
            print(format_comparison(last_report), end="")
            status(f"dense-only done — see {args.output_dir / 'dense_only_table.md'}")
        if args.stage in ("hybrid-bm25", "both"):
            last_report = run_hybrid_bm25_comparison(
                models=models,
                golden_path=args.golden,
                chunks_path=args.chunks,
                output_dir=args.output_dir,
                max_chunks=args.max_chunks,
                candidate_limit=args.candidate_limit,
                fusion_pool=args.fusion_pool,
                rrf_k=args.rrf_k,
                batch_size=args.batch_size,
                force_reembed=args.force_reembed,
                require_dense_cache=args.stage == "hybrid-bm25",
                progress=status,
            )
            print(format_comparison(last_report), end="")
            status(f"hybrid done — see {args.output_dir / 'hybrid_bm25_table.md'}")
        if args.stage == "bge-sparse":
            try:
                sparse_provider = BgeM3EmbeddingProvider(
                    device="cpu",
                    batch_size=args.sparse_batch_size,
                    use_fp16=False,
                    enable_sparse=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "failed to load BGE-M3 for learned-sparse (need shamela-rag[bge]): %s",
                    exc,
                )
                return 1
            last_report = run_bge_sparse_ablation(
                dense_provider=models["bge-m3"],
                sparse_provider=sparse_provider,
                golden_path=args.golden,
                chunks_path=args.chunks,
                output_dir=args.output_dir,
                max_chunks=args.max_chunks,
                candidate_limit=args.candidate_limit,
                fusion_pool=args.fusion_pool,
                rrf_k=args.rrf_k,
                dense_batch_size=args.batch_size,
                sparse_batch_size=args.sparse_batch_size,
                force_reembed=args.force_reembed,
                require_dense_cache=not args.force_reembed,
                progress=status,
            )
            print(format_comparison(last_report), end="")
            status(f"bge-sparse done — see {args.output_dir / 'bge_sparse_table.md'}")
    except Exception as exc:  # noqa: BLE001
        logger.error("comparison failed: %s", exc)
        return 1

    return 0


def run_ask(args: argparse.Namespace) -> int:
    import json as _json

    from shamela_rag.factory import build_general_qa_service
    from shamela_rag.retrieval.filters import RetrievalFilter

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    reranker = None
    if args.no_rerank:
        from shamela_rag.retrieval.rerank import LexicalOverlapReranker

        reranker = LexicalOverlapReranker()

    service = build_general_qa_service(model=args.model or "bge-m3", reranker=reranker)
    filters = None
    if args.book is not None or args.category is not None:
        filters = RetrievalFilter(book_id=args.book, category_id=args.category)

    answer = service.answer(args.question, k=args.k, filters=filters)
    if args.json:
        print(
            _json.dumps(
                {
                    "answer": answer.text,
                    "deflected": answer.deflected,
                    "citations": [
                        {
                            "marker": c.marker,
                            "id": c.id,
                            "chunk_id": c.chunk_id,
                            "book_title": c.book_title,
                            "author": c.author,
                            "page": c.page,
                            "content_role": c.content_role,
                        }
                        for c in answer.citations
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(answer.text)
    for citation in answer.citations:
        footnote = " [footnote]" if citation.content_role == "footnote" else ""
        print(
            f"[{citation.marker}] {citation.book_title} - {citation.author} "
            f"(p.{citation.page}){footnote}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args, _build_service(args.model))
    if args.command == "validate-structure":
        return run_validate_structure(args)
    if args.command == "audit-structure":
        return run_audit_structure(args)
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "build-bm25":
        return run_build_bm25(args)
    if args.command == "ask":
        return run_ask(args)
    if args.command == "compare-qwen-quant":
        return run_compare_qwen_quant(args)
    if args.command == "compare-dense-models":
        return run_compare_dense_models(args)
    parser.error(f"unknown command: {args.command}")  # required subparser makes this unreachable
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
