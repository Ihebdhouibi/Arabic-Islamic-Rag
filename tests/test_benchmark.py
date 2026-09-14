"""Unit tests for the sizing/latency benchmark maths and report (issue #148).

The measurement functions themselves need live stores, so what is covered here is the part that
has to be right for the report to be trustworthy: percentiles, the extrapolation arithmetic, and
that the rendered report states its numbers rather than implying them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shamela_rag.data.discovery import BookLocation
from shamela_rag.eval.benchmark import (
    FULL_CORPUS_PAGES,
    BenchmarkReport,
    LatencyReport,
    PostgresSizing,
    Projection,
    QdrantSizing,
    StageLatency,
    TableSize,
    _percentile,
    count_pages_jsonl,
    format_benchmark_report,
    load_book_ids_file,
    project_full_corpus,
    resolve_book_ids,
    sample_books_for_benchmark,
)

_GB = 1024**3


def _postgres(pages: int = 1000, books: int = 40, chunks: int = 2500) -> PostgresSizing:
    return PostgresSizing(
        tables=(TableSize("chunks", 800_000_000), TableSize("books", 200_000)),
        database_bytes=_GB,
        books=books,
        pages=pages,
        chunks=chunks,
    )


def _qdrant(points: int = 2500, disk: int | None = _GB // 2) -> QdrantSizing:
    return QdrantSizing(
        collection="shamela_general", points=points, disk_bytes=disk, vector_dim=1024
    )


def _write_book(root: Path, category_id: int, book_id: int, pages: int) -> BookLocation:
    book_dir = root / f"{category_id:02d}__cat" / f"{book_id}__book"
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "pages.jsonl").write_text(
        "\n".join(f'{{"page_id": {i}}}' for i in range(1, pages + 1)) + "\n",
        encoding="utf-8",
    )
    (book_dir / "toc.jsonl").write_text("{}\n", encoding="utf-8")
    (book_dir / "book_metadata.json").write_text("{}", encoding="utf-8")
    return BookLocation(
        book_dir=book_dir, book_id=book_id, category_id=category_id, has_all_files=True
    )


def test_percentile_is_nearest_rank() -> None:
    samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert _percentile(samples, 95) == 100
    assert _percentile(samples, 50) == 50
    assert _percentile([], 95) == 0.0
    assert _percentile([7], 95) == 7


def test_percentile_ignores_input_order() -> None:
    assert _percentile([50, 10, 90, 30], 95) == _percentile([90, 30, 10, 50], 95)


def test_projection_scales_by_measured_ratio() -> None:
    proj = Projection(
        label="Postgres database (GB)",
        measured_value=2.0,
        measured_units=1000,
        projected_units=1_000_000,
    )
    assert proj.per_unit == 2.0 / 1000
    assert proj.projected == 2000.0


def test_projection_of_zero_sample_does_not_divide_by_zero() -> None:
    proj = Projection(label="x", measured_value=5.0, measured_units=0, projected_units=100)
    assert proj.per_unit == 0.0
    assert proj.projected == 0.0


def test_full_corpus_projection_uses_page_ratio() -> None:
    pg = _postgres(pages=1000)
    projections = project_full_corpus(pg, _qdrant())
    by_label = {p.label: p for p in projections}

    postgres_projection = by_label["Postgres database (GB)"]
    assert postgres_projection.projected_units == FULL_CORPUS_PAGES
    assert postgres_projection.projected == 1.0 / 1000 * FULL_CORPUS_PAGES


def test_projection_omits_qdrant_disk_when_unmeasured() -> None:
    labels = {p.label for p in project_full_corpus(_postgres(), _qdrant(disk=None))}
    assert "Qdrant on disk (GB)" not in labels
    assert "Qdrant points" in labels


def test_sizing_ratios() -> None:
    pg = _postgres(pages=1000, books=40)
    assert pg.bytes_per_page == _GB / 1000
    assert pg.bytes_per_book == _GB / 40
    assert pg.database_gb == 1.0


def test_qdrant_bytes_per_point_is_none_without_disk() -> None:
    assert _qdrant(disk=None).bytes_per_point is None
    assert _qdrant(points=0).bytes_per_point is None
    assert _qdrant(points=100, disk=1000).bytes_per_point == 10.0


def test_latency_within_budget_flag() -> None:
    fast = LatencyReport(
        query_count=3,
        total=StageLatency("total", (100, 200, 300)),
        stages=(),
        budget_ms=8000,
    )
    slow = LatencyReport(
        query_count=3,
        total=StageLatency("total", (9000, 9500, 10000)),
        stages=(),
        budget_ms=8000,
    )
    assert fast.within_budget is True
    assert slow.within_budget is False


def test_report_renders_measured_and_projected_sections() -> None:
    report = BenchmarkReport(
        postgres=_postgres(),
        qdrant=_qdrant(),
        latency=LatencyReport(
            query_count=10,
            total=StageLatency("total", (500, 600, 700)),
            stages=(
                StageLatency("dense", (100, 120, 140)),
                StageLatency("rerank", (300, 350, 400)),
            ),
            budget_ms=8000,
        ),
        sample_categories=(1, 2, 3),
        notes=("Sample ingested with 1 book per category.",),
    )
    markdown = format_benchmark_report(report)

    assert "# Sizing and latency benchmark" in markdown
    assert "## Sample ingested" in markdown
    assert "## Measured storage" in markdown
    assert "## Full-corpus projection" in markdown
    assert "## Retrieval latency" in markdown
    assert "projected = measured / sample_pages" in markdown
    assert "dense" in markdown and "rerank" in markdown
    assert "Categories covered: 1, 2, 3" in markdown
    assert "within the 8,000 ms budget" in markdown


def test_report_is_explicit_when_qdrant_disk_is_unmeasured() -> None:
    markdown = format_benchmark_report(
        BenchmarkReport(postgres=_postgres(), qdrant=_qdrant(disk=None))
    )
    assert "not measured" in markdown


def test_report_flags_an_over_budget_p95() -> None:
    markdown = format_benchmark_report(
        BenchmarkReport(
            postgres=_postgres(),
            qdrant=_qdrant(),
            latency=LatencyReport(
                query_count=2,
                total=StageLatency("total", (12000, 15000)),
                stages=(),
                budget_ms=8000,
            ),
        )
    )
    assert "OVER the 8,000 ms budget" in markdown


def test_report_lists_failed_queries() -> None:
    markdown = format_benchmark_report(
        BenchmarkReport(
            postgres=_postgres(),
            qdrant=_qdrant(),
            latency=LatencyReport(
                query_count=1,
                total=StageLatency("total", (100,)),
                stages=(),
                budget_ms=8000,
                errors=("bad query: boom",),
            ),
        )
    )
    assert "Queries that errored" in markdown
    assert "bad query: boom" in markdown


def test_count_pages_jsonl(tmp_path: Path) -> None:
    loc = _write_book(tmp_path, category_id=1, book_id=10, pages=7)
    assert count_pages_jsonl(loc.book_dir) == 7


def test_load_book_ids_file(tmp_path: Path) -> None:
    path = tmp_path / "books.txt"
    path.write_text("# sample\n101\n\n202 extra\n# end\n", encoding="utf-8")
    assert load_book_ids_file(path) == [101, 202]


def test_load_book_ids_file_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no book ids"):
        load_book_ids_file(path)


def test_resolve_book_ids(tmp_path: Path) -> None:
    _write_book(tmp_path, 1, 10, pages=3)
    _write_book(tmp_path, 2, 20, pages=5)
    resolved = resolve_book_ids(tmp_path, [20, 10, 10])
    assert [loc.book_id for loc in resolved] == [20, 10]


def test_resolve_book_ids_missing_raises(tmp_path: Path) -> None:
    _write_book(tmp_path, 1, 10, pages=3)
    with pytest.raises(ValueError, match="not found"):
        resolve_book_ids(tmp_path, [10, 999])


def test_sample_books_for_benchmark_covers_categories_and_target(tmp_path: Path) -> None:
    _write_book(tmp_path, 1, 11, pages=2)
    _write_book(tmp_path, 1, 12, pages=50)
    _write_book(tmp_path, 2, 21, pages=3)
    _write_book(tmp_path, 2, 22, pages=80)
    _write_book(tmp_path, 3, 31, pages=4)
    _write_book(tmp_path, 3, 32, pages=100)

    picked = sample_books_for_benchmark(tmp_path, target_books=5)
    assert len(picked) == 5
    categories = {loc.category_id for loc in picked}
    assert categories == {1, 2, 3}
    assert {11, 21, 31}.issubset({loc.book_id for loc in picked})


def test_sample_books_for_benchmark_rejects_non_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_books"):
        sample_books_for_benchmark(tmp_path, target_books=0)
