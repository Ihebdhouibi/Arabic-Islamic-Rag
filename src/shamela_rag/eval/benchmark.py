"""Sizing and latency benchmark over an ingested sample (issue #148).

Two questions this answers with measurements rather than formulas:

1. How large do Postgres and Qdrant get for the full corpus? Measured on a stratified sample, then
   extrapolated per page and per book, with the ratios shown so the projection is reproducible.
2. What does retrieval latency look like, split by stage? Measured by running a query set through
   the real pipeline and reading the per-stage timings the retrieval service reports.

Every number in the emitted report is either measured here or derived from a measured ratio; the
report states which.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

# Full-corpus totals, counted from the dataset (see docs/technical_docs).
FULL_CORPUS_BOOKS = 8_589
FULL_CORPUS_PAGES = 7_611_186

_BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class TableSize:
    name: str
    total_bytes: int

    @property
    def gb(self) -> float:
        return self.total_bytes / _BYTES_PER_GB


@dataclass(frozen=True)
class PostgresSizing:
    tables: tuple[TableSize, ...]
    database_bytes: int
    books: int
    pages: int
    chunks: int

    @property
    def database_gb(self) -> float:
        return self.database_bytes / _BYTES_PER_GB

    @property
    def bytes_per_page(self) -> float:
        return self.database_bytes / self.pages if self.pages else 0.0

    @property
    def bytes_per_book(self) -> float:
        return self.database_bytes / self.books if self.books else 0.0


@dataclass(frozen=True)
class QdrantSizing:
    collection: str
    points: int
    disk_bytes: int | None  # None when the storage path is not reachable from here
    vector_dim: int

    @property
    def disk_gb(self) -> float | None:
        return self.disk_bytes / _BYTES_PER_GB if self.disk_bytes is not None else None

    @property
    def bytes_per_point(self) -> float | None:
        if self.disk_bytes is None or not self.points:
            return None
        return self.disk_bytes / self.points


@dataclass(frozen=True)
class Projection:
    """A full-corpus number derived from a measured per-unit ratio."""

    label: str
    measured_value: float
    measured_units: int
    projected_units: int

    @property
    def per_unit(self) -> float:
        return self.measured_value / self.measured_units if self.measured_units else 0.0

    @property
    def projected(self) -> float:
        return self.per_unit * self.projected_units


@dataclass(frozen=True)
class StageLatency:
    stage: str
    samples: tuple[int, ...]

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return _percentile(self.samples, 95)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples) if self.samples else 0.0


@dataclass(frozen=True)
class LatencyReport:
    query_count: int
    total: StageLatency
    stages: tuple[StageLatency, ...]
    budget_ms: int
    errors: tuple[str, ...] = ()

    @property
    def within_budget(self) -> bool:
        return self.total.p95 <= self.budget_ms


@dataclass(frozen=True)
class BenchmarkReport:
    postgres: PostgresSizing
    qdrant: QdrantSizing
    latency: LatencyReport | None = None
    sample_categories: tuple[int, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def _percentile(samples: Sequence[int], pct: int) -> float:
    """Nearest-rank percentile; stable for the small sample sizes a benchmark run produces."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    return float(ordered[min(rank, len(ordered)) - 1])


def measure_postgres(engine: Engine) -> PostgresSizing:
    """Real on-disk size per table plus the row counts the ratios are computed against."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT relname, pg_total_relation_size(c.oid) AS total_bytes
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relkind = 'r'
                ORDER BY total_bytes DESC
                """
            )
        ).all()
        database_bytes = conn.execute(
            text("SELECT pg_database_size(current_database())")
        ).scalar_one()
        books = conn.execute(text("SELECT count(*) FROM books")).scalar_one()
        chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        pages = conn.execute(
            text("SELECT count(DISTINCT (book_id, start_page_id)) FROM chunks")
        ).scalar_one()

    return PostgresSizing(
        tables=tuple(TableSize(name=r[0], total_bytes=int(r[1])) for r in rows),
        database_bytes=int(database_bytes),
        books=int(books),
        pages=int(pages),
        chunks=int(chunks),
    )


def measure_qdrant(client, collection: str, *, storage_path: Path | None = None) -> QdrantSizing:
    """Point count from the API; disk size from the storage directory when it is reachable."""
    info = client.get_collection(collection)
    points = int(info.points_count or 0)

    vectors = getattr(getattr(info.config, "params", None), "vectors", None)
    dim = 0
    if isinstance(vectors, dict):
        for spec in vectors.values():
            dim = max(dim, int(getattr(spec, "size", 0) or 0))
    else:
        dim = int(getattr(vectors, "size", 0) or 0)

    disk_bytes = _directory_size(storage_path) if storage_path is not None else None
    return QdrantSizing(collection=collection, points=points, disk_bytes=disk_bytes, vector_dim=dim)


def docker_directory_size(container: str, path: str = "/qdrant/storage") -> int | None:
    """Byte size of a directory inside a running container.

    Qdrant keeps its storage in a named volume, so there is no host path to walk; ``du`` inside
    the container is the honest measurement. Returns None when docker is unavailable.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "exec", container, "du", "-sb", path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return int(result.stdout.split()[0])
    except (ValueError, IndexError):
        return None


def _directory_size(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def project_full_corpus(postgres: PostgresSizing, qdrant: QdrantSizing) -> tuple[Projection, ...]:
    """Scale each measured total by its own measured per-page ratio."""
    projections = [
        Projection(
            label="Postgres database (GB)",
            measured_value=postgres.database_gb,
            measured_units=postgres.pages,
            projected_units=FULL_CORPUS_PAGES,
        ),
        Projection(
            label="Chunk rows",
            measured_value=float(postgres.chunks),
            measured_units=postgres.pages,
            projected_units=FULL_CORPUS_PAGES,
        ),
        Projection(
            label="Qdrant points",
            measured_value=float(qdrant.points),
            measured_units=postgres.pages,
            projected_units=FULL_CORPUS_PAGES,
        ),
    ]
    if qdrant.disk_gb is not None:
        projections.append(
            Projection(
                label="Qdrant on disk (GB)",
                measured_value=qdrant.disk_gb,
                measured_units=postgres.pages,
                projected_units=FULL_CORPUS_PAGES,
            )
        )
    return tuple(projections)


def measure_latency(
    service,
    queries: Sequence[str],
    *,
    budget_ms: int,
    top_k: int = 10,
    warmup: int = 1,
) -> LatencyReport:
    """Run the query set through the real pipeline, collecting per-stage timings.

    The first ``warmup`` queries are discarded so model load and connection setup do not land in
    the reported percentiles.
    """
    from shamela_rag.retrieval.expand import ExpandMode, ExpansionConfig
    from shamela_rag.retrieval.service import RetrievalConfig

    config = RetrievalConfig(
        final_k=top_k,
        rerank_top_k=top_k,
        expansion=ExpansionConfig(mode=ExpandMode.NONE),
    )

    totals: list[int] = []
    per_stage: dict[str, list[int]] = {}
    errors: list[str] = []

    for index, query in enumerate(queries):
        try:
            outcome = service.retrieve_with_outcome(query, k=top_k, config=config)
        except Exception as exc:  # noqa: BLE001 - a failed query is data, not a crash
            errors.append(f"{query[:40]}: {exc}")
            continue
        if index < warmup:
            continue
        totals.append(outcome.elapsed_ms)
        for stage, ms in outcome.stage_ms.items():
            per_stage.setdefault(stage, []).append(ms)

    stage_order = ["translate", "dense", "sparse", "fuse", "hydrate", "rerank", "expand"]
    stages = tuple(
        StageLatency(stage=name, samples=tuple(per_stage[name]))
        for name in stage_order
        if name in per_stage
    )
    return LatencyReport(
        query_count=len(totals),
        total=StageLatency(stage="total", samples=tuple(totals)),
        stages=stages,
        budget_ms=budget_ms,
        errors=tuple(errors),
    )


def format_benchmark_report(report: BenchmarkReport) -> str:
    pg = report.postgres
    qd = report.qdrant
    lines = [
        "# Sizing and latency benchmark",
        "",
        "Measured on a stratified corpus sample. Full-corpus figures are projections from the",
        f"measured per-page ratios below, against {FULL_CORPUS_BOOKS:,} books and",
        f"{FULL_CORPUS_PAGES:,} pages.",
        "",
        "## Sample ingested",
        "",
        f"- Books: **{pg.books:,}**",
        f"- Pages (distinct book/page pairs in chunks): **{pg.pages:,}**",
        f"- Chunk rows: **{pg.chunks:,}**",
        f"- Qdrant points: **{qd.points:,}** (dim {qd.vector_dim})",
    ]
    if report.sample_categories:
        covered = ", ".join(str(c) for c in report.sample_categories)
        lines.append(f"- Categories covered: {covered}")

    lines += [
        "",
        "## Measured storage",
        "",
        "| Store | Measured | Per page | Per book |",
        "| --- | ---: | ---: | ---: |",
        f"| Postgres (database) | {pg.database_gb:.3f} GB | "
        f"{pg.bytes_per_page / 1024:.1f} KB | {pg.bytes_per_book / 1024:.1f} KB |",
    ]
    if qd.disk_gb is not None:
        per_page = qd.disk_bytes / pg.pages / 1024 if pg.pages else 0.0
        per_book = qd.disk_bytes / pg.books / 1024 if pg.books else 0.0
        lines.append(
            f"| Qdrant (storage dir) | {qd.disk_gb:.3f} GB | {per_page:.1f} KB | "
            f"{per_book:.1f} KB |"
        )
    else:
        lines.append("| Qdrant (storage dir) | not measured | - | - |")

    lines += ["", "### Postgres by table", "", "| Table | Size |", "| --- | ---: |"]
    for table in pg.tables:
        lines.append(f"| {table.name} | {_human_bytes(table.total_bytes)} |")

    lines += [
        "",
        "## Full-corpus projection",
        "",
        "Each row scales the measured value by its own per-page ratio, so the arithmetic is",
        "visible and re-checkable: `projected = measured / sample_pages * 7,611,186`.",
        "",
        "| Quantity | Measured | Per page | Projected (full corpus) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for proj in project_full_corpus(pg, qd):
        lines.append(
            f"| {proj.label} | {proj.measured_value:,.3f} | {proj.per_unit:.6f} | "
            f"{proj.projected:,.1f} |"
        )

    if report.latency is not None:
        lat = report.latency
        lines += [
            "",
            "## Retrieval latency",
            "",
            f"{lat.query_count} timed queries (warm-up excluded), budget {lat.budget_ms:,} ms.",
            "",
            "| Stage | p50 (ms) | p95 (ms) | mean (ms) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for stage in lat.stages:
            lines.append(
                f"| {stage.stage} | {stage.p50:,.0f} | {stage.p95:,.0f} | {stage.mean:,.0f} |"
            )
        lines.append(
            f"| **total** | **{lat.total.p50:,.0f}** | **{lat.total.p95:,.0f}** | "
            f"**{lat.total.mean:,.0f}** |"
        )
        verdict = "within" if lat.within_budget else "OVER"
        lines += [
            "",
            f"p95 total is **{lat.total.p95:,.0f} ms**, {verdict} the {lat.budget_ms:,} ms budget.",
        ]
        if lat.errors:
            lines += ["", "Queries that errored:", ""]
            lines += [f"- {err}" for err in lat.errors]

    if report.notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in report.notes]

    lines.append("")
    return "\n".join(lines)


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"
