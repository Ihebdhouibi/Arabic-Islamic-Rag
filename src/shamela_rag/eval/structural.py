"""Structural validation harness for the chunking pipeline (M6-06).

Runs ``chunk_book`` over books and checks hard invariants: verbatim offsets, source coverage,
determinism, inline ``toc-N`` boundaries, section integrity, within-section overlap, and footnote
roles. Ambiguous / low-confidence boundaries are reported as soft findings and do not fail the gate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from shamela_rag.chunking.boundaries import (
    Boundary,
    BoundarySource,
    boundary_source_counts,
    detect_page_boundaries,
)
from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.navigation import DEFAULT_MIN_CONTENT_TOKENS, is_navigational
from shamela_rag.chunking.orchestrator import BookChunk, ChunkingResult, chunk_book
from shamela_rag.chunking.sizing import DEFAULT_POLICY, SizePolicy
from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.data.discovery import BookLocation, iter_valid_books
from shamela_rag.data.models import Page, TocEntry, load_book, load_pages, load_toc
from shamela_rag.data.ordering import order_pages

ProgressFn = Callable[[str], None]


def _log_progress(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


class Severity(StrEnum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    check: str
    message: str
    page_id: int | None = None


@dataclass
class BookReport:
    book_id: int
    book_dir: Path
    chunk_count: int
    category_id: int | None = None
    findings: list[Finding] = field(default_factory=list)
    confidence_counts: dict[str, int] = field(default_factory=dict)
    boundary_source_counts: dict[str, int] = field(default_factory=dict)
    ambiguous_boundaries: int = 0
    heading_recovery_candidates: int = 0

    @property
    def hard_violations(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.HARD]

    @property
    def ok(self) -> bool:
        return not self.hard_violations


@dataclass(frozen=True)
class CorpusReport:
    books: tuple[BookReport, ...]

    @property
    def hard_violation_count(self) -> int:
        return sum(len(book.hard_violations) for book in self.books)

    @property
    def ok(self) -> bool:
        return bool(self.books) and self.hard_violation_count == 0


@dataclass(frozen=True)
class CategoryAuditRow:
    category_id: int
    book_count: int
    boundary_source_counts: dict[str, int]
    confidence_counts: dict[str, int]
    ambiguous_boundaries: int
    low_confidence_boundaries: int

    @property
    def total_boundaries(self) -> int:
        return sum(self.boundary_source_counts.values())

    def boundary_pct(self, source: str) -> float:
        total = self.total_boundaries
        if total == 0:
            return 0.0
        return 100.0 * self.boundary_source_counts.get(source, 0) / total


@dataclass(frozen=True)
class CategoryAuditReport:
    rows: tuple[CategoryAuditRow, ...]
    books_per_category: int

    @property
    def category_count(self) -> int:
        return len(self.rows)


def _hard(check: str, message: str, page_id: int | None = None) -> Finding:
    return Finding(Severity.HARD, check, message, page_id)


def _soft(check: str, message: str, page_id: int | None = None) -> Finding:
    return Finding(Severity.SOFT, check, message, page_id)


def _segment_ranges(body: str, boundaries: Sequence[Boundary]) -> list[tuple[int, int]]:
    ordered = sorted(boundaries, key=lambda b: b.offset)
    points: list[int] = []
    if not ordered or ordered[0].offset > 0:
        points.append(0)
    points.extend(b.offset for b in ordered)

    ranges: list[tuple[int, int]] = []
    for index, start in enumerate(points):
        end = points[index + 1] if index + 1 < len(points) else len(body)
        if end > start:
            ranges.append((start, end))
    return ranges


def _toc_by_page(toc: Sequence[TocEntry]) -> dict[int, list[TocEntry]]:
    by_page: dict[int, list[TocEntry]] = {}
    for entry in toc:
        by_page.setdefault(entry.page_id, []).append(entry)
    return by_page


def _count_boundary_sources(pages: Sequence[Page], toc: Sequence[TocEntry]) -> dict[str, int]:
    by_page = _toc_by_page(toc)
    counts: dict[str, int] = {}
    for page in pages:
        boundaries = detect_page_boundaries(page.body, by_page.get(page.page_id, []))
        for source, count in boundary_source_counts(boundaries).items():
            counts[source.value] = counts.get(source.value, 0) + count
    return counts


def stratified_book_locations(
    corpus_root: Path,
    *,
    books_per_category: int = 1,
) -> list[BookLocation]:
    by_category: dict[int, list[BookLocation]] = defaultdict(list)
    for location in iter_valid_books(corpus_root):
        if location.category_id is not None:
            by_category[location.category_id].append(location)
    selected: list[BookLocation] = []
    for category_id in sorted(by_category):
        selected.extend(by_category[category_id][:books_per_category])
    return selected


def _check_verbatim(chunks: Sequence[BookChunk], pages: dict[int, Page]) -> list[Finding]:
    findings: list[Finding] = []
    for chunk in chunks:
        page = pages.get(chunk.page_id)
        if page is None:
            findings.append(_hard("verbatim", f"chunk references missing page_id={chunk.page_id}"))
            continue
        haystack = page.footnotes if chunk.content_role is ContentRole.FOOTNOTE else page.body
        if haystack is None:
            findings.append(
                _hard(
                    "verbatim",
                    f"chunk role={chunk.content_role} but page has no matching text",
                    chunk.page_id,
                )
            )
            continue
        if not (0 <= chunk.start_offset <= chunk.end_offset <= len(haystack)):
            findings.append(
                _hard(
                    "verbatim",
                    f"offsets [{chunk.start_offset}:{chunk.end_offset}] out of range "
                    f"(len={len(haystack)})",
                    chunk.page_id,
                )
            )
            continue
        if haystack[chunk.start_offset : chunk.end_offset] != chunk.source_text:
            findings.append(
                _hard(
                    "verbatim",
                    "source_text does not equal haystack[start:end]",
                    chunk.page_id,
                )
            )
    return findings


def _check_determinism(
    book_dir: Path, policy: SizePolicy, min_content_tokens: int
) -> list[Finding]:
    first = chunk_book(book_dir, policy=policy, min_content_tokens=min_content_tokens)
    second = chunk_book(book_dir, policy=policy, min_content_tokens=min_content_tokens)
    if first != second:
        return [_hard("determinism", "chunk_book produced different results on two runs")]
    return []


def _check_inline_toc(pages: Sequence[Page], toc: Sequence[TocEntry]) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    ambiguous = 0
    by_page = _toc_by_page(toc)

    for page in pages:
        toc_on_page = by_page.get(page.page_id, [])
        boundaries = detect_page_boundaries(page.body, toc_on_page)
        ambiguous += sum(1 for b in boundaries if b.source is BoundarySource.AMBIGUOUS_TOC_PAGE)
        by_offset = {b.offset: b for b in boundaries if b.source is BoundarySource.INLINE_TOC}

        for span in parse_title_spans(page.body):
            boundary = by_offset.get(span.start)
            if boundary is None:
                findings.append(
                    _hard(
                        "inline_toc",
                        f"toc-{span.shamela_title_id} at offset {span.start} has no boundary",
                        page.page_id,
                    )
                )
                continue
            if boundary.shamela_title_id != span.shamela_title_id:
                findings.append(
                    _hard(
                        "toc_n_mapping",
                        f"toc-{span.shamela_title_id} mapped to "
                        f"shamela_title_id={boundary.shamela_title_id}",
                        page.page_id,
                    )
                )
    return findings, ambiguous


def _content_ranges(
    page: Page, toc_on_page: Sequence[TocEntry], *, min_content_tokens: int
) -> list[tuple[int, int]]:
    boundaries = detect_page_boundaries(page.body, toc_on_page)
    return [
        (start, end)
        for start, end in _segment_ranges(page.body, boundaries)
        if not is_navigational(page.body[start:end], min_content_tokens=min_content_tokens)
    ]


def _check_section_integrity(
    chunks: Sequence[BookChunk],
    pages: Sequence[Page],
    toc: Sequence[TocEntry],
    *,
    min_content_tokens: int,
) -> list[Finding]:
    findings: list[Finding] = []
    by_page = _toc_by_page(toc)
    page_ranges = {
        page.page_id: _content_ranges(
            page, by_page.get(page.page_id, []), min_content_tokens=min_content_tokens
        )
        for page in pages
    }

    for chunk in chunks:
        if chunk.content_role is not ContentRole.BODY:
            continue
        ranges = page_ranges.get(chunk.page_id, [])
        if not any(
            start <= chunk.start_offset and chunk.end_offset <= end for start, end in ranges
        ):
            findings.append(
                _hard(
                    "section_integrity",
                    f"chunk [{chunk.start_offset}:{chunk.end_offset}] crosses or leaves "
                    "its structural section",
                    chunk.page_id,
                )
            )
    return findings


def _check_overlap_within_section(chunks: Sequence[BookChunk]) -> list[Finding]:
    findings: list[Finding] = []
    by_page: dict[int, list[BookChunk]] = {}
    for chunk in chunks:
        if chunk.content_role is ContentRole.BODY:
            by_page.setdefault(chunk.page_id, []).append(chunk)

    for page_id, group in by_page.items():
        ordered = sorted(group, key=lambda c: (c.start_offset, c.end_offset))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if right.start_offset >= left.end_offset:
                    break
                if left.trail != right.trail:
                    findings.append(
                        _hard(
                            "overlap_scope",
                            "overlapping chunks have different trails "
                            f"({left.trail!r} vs {right.trail!r})",
                            page_id,
                        )
                    )
    return findings


def _check_footnotes(chunks: Sequence[BookChunk]) -> list[Finding]:
    findings: list[Finding] = []
    for chunk in chunks:
        if chunk.content_role not in (ContentRole.BODY, ContentRole.FOOTNOTE):
            findings.append(
                _hard(
                    "footnotes",
                    f"unknown content_role={chunk.content_role!r}",
                    chunk.page_id,
                )
            )
    return findings


def _mark_range(mask: list[bool], start: int, end: int) -> None:
    end = min(end, len(mask))
    start = max(0, start)
    for i in range(start, end):
        mask[i] = True


def _check_coverage(
    chunks: Sequence[BookChunk],
    pages: Sequence[Page],
    toc: Sequence[TocEntry],
    *,
    min_content_tokens: int,
) -> list[Finding]:
    findings: list[Finding] = []
    by_page = _toc_by_page(toc)
    body_chunks: dict[int, list[BookChunk]] = {}
    footnote_chunks: dict[int, list[BookChunk]] = {}
    for chunk in chunks:
        target = footnote_chunks if chunk.content_role is ContentRole.FOOTNOTE else body_chunks
        target.setdefault(chunk.page_id, []).append(chunk)

    for page in pages:
        covered = [False] * len(page.body)
        ignored = [False] * len(page.body)
        boundaries = detect_page_boundaries(page.body, by_page.get(page.page_id, []))
        for start, end in _segment_ranges(page.body, boundaries):
            if is_navigational(page.body[start:end], min_content_tokens=min_content_tokens):
                _mark_range(ignored, start, end)
        for chunk in body_chunks.get(page.page_id, []):
            _mark_range(covered, chunk.start_offset, chunk.end_offset)

        uncovered = [
            ch
            for i, ch in enumerate(page.body)
            if not covered[i] and not ignored[i] and not ch.isspace()
        ]
        if uncovered:
            sample = "".join(uncovered[:40]).replace("\n", "\\n")
            findings.append(
                _hard(
                    "coverage",
                    f"{len(uncovered)} body char(s) neither chunked nor ignored "
                    f"(sample={sample!r})",
                    page.page_id,
                )
            )

        if page.footnotes:
            covered_fn = [False] * len(page.footnotes)
            for chunk in footnote_chunks.get(page.page_id, []):
                _mark_range(covered_fn, chunk.start_offset, chunk.end_offset)
            uncovered_fn = [
                ch for i, ch in enumerate(page.footnotes) if not covered_fn[i] and not ch.isspace()
            ]
            if uncovered_fn:
                sample = "".join(uncovered_fn[:40]).replace("\n", "\\n")
                findings.append(
                    _hard(
                        "coverage",
                        f"{len(uncovered_fn)} footnote char(s) not covered (sample={sample!r})",
                        page.page_id,
                    )
                )
    return findings


def _soft_boundary_report(result: ChunkingResult, ambiguous: int) -> list[Finding]:
    findings: list[Finding] = []
    low = result.stats.confidence_counts.get("low", 0)
    if ambiguous:
        findings.append(
            _soft(
                "ambiguous_boundaries",
                f"{ambiguous} ambiguous_toc_page boundary(ies)",
            )
        )
    if low:
        findings.append(_soft("low_confidence", f"{low} low-confidence boundary(ies)"))
    if result.stats.heading_recovery_candidates:
        findings.append(
            _soft(
                "heading_recovery",
                f"{result.stats.heading_recovery_candidates} heading-recovery candidate(s)",
            )
        )
    return findings


def validate_book(
    book_dir: Path,
    *,
    policy: SizePolicy = DEFAULT_POLICY,
    min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS,
) -> BookReport:
    book = load_book(book_dir)
    pages = list(order_pages(load_pages(book_dir)))
    toc = list(load_toc(book_dir))
    result = chunk_book(book_dir, policy=policy, min_content_tokens=min_content_tokens)
    pages_by_id = {p.page_id: p for p in pages}

    findings: list[Finding] = []
    findings.extend(_check_determinism(book_dir, policy, min_content_tokens))
    findings.extend(_check_verbatim(result.chunks, pages_by_id))
    toc_findings, ambiguous = _check_inline_toc(pages, toc)
    findings.extend(toc_findings)
    findings.extend(
        _check_section_integrity(result.chunks, pages, toc, min_content_tokens=min_content_tokens)
    )
    findings.extend(_check_overlap_within_section(result.chunks))
    findings.extend(_check_footnotes(result.chunks))
    findings.extend(
        _check_coverage(result.chunks, pages, toc, min_content_tokens=min_content_tokens)
    )
    findings.extend(_soft_boundary_report(result, ambiguous))

    return BookReport(
        book_id=book.book_id,
        book_dir=book_dir,
        chunk_count=len(result.chunks),
        category_id=book.category_id,
        findings=findings,
        confidence_counts=dict(result.stats.confidence_counts),
        boundary_source_counts=_count_boundary_sources(pages, toc),
        ambiguous_boundaries=ambiguous,
        heading_recovery_candidates=result.stats.heading_recovery_candidates,
    )


def validate_corpus(
    corpus_root: Path,
    *,
    limit: int | None = 20,
    book_id: int | None = None,
    category_id: int | None = None,
    stratified: bool = False,
    books_per_category: int = 1,
    policy: SizePolicy = DEFAULT_POLICY,
    min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS,
    progress: ProgressFn | None = None,
) -> CorpusReport:
    if stratified:
        _log_progress(progress, "scanning corpus for stratified sample...")
        locations = stratified_book_locations(corpus_root, books_per_category=books_per_category)
        if book_id is not None:
            locations = [loc for loc in locations if loc.book_id == book_id]
        if category_id is not None:
            locations = [loc for loc in locations if loc.category_id == category_id]
        categories = {loc.category_id for loc in locations}
        _log_progress(
            progress,
            (
                f"selected {len(locations)} book(s) across {len(categories)} "
                f"categor{'y' if len(categories) == 1 else 'ies'} "
                f"({books_per_category} per category)"
            ),
        )
    else:
        locations = list(iter_valid_books(corpus_root))
        if book_id is not None:
            locations = [loc for loc in locations if loc.book_id == book_id]
        if category_id is not None:
            locations = [loc for loc in locations if loc.category_id == category_id]
        if limit is not None:
            locations = locations[:limit]

    total = len(locations)
    reports: list[BookReport] = []
    for index, loc in enumerate(locations, start=1):
        _log_progress(
            progress,
            f"[{index}/{total}] validating category={loc.category_id} book_id={loc.book_id}",
        )
        reports.append(
            validate_book(loc.book_dir, policy=policy, min_content_tokens=min_content_tokens)
        )
        book_report = reports[-1]
        inline_toc = book_report.boundary_source_counts.get(BoundarySource.INLINE_TOC.value, 0)
        weak = (
            book_report.boundary_source_counts.get(BoundarySource.PARAGRAPH_FALLBACK.value, 0)
            + book_report.boundary_source_counts.get(BoundarySource.AMBIGUOUS_TOC_PAGE.value, 0)
            + book_report.boundary_source_counts.get(BoundarySource.TOC_PAGE_FALLBACK.value, 0)
        )
        status = "PASS" if book_report.ok else "FAIL"
        _log_progress(
            progress,
            (
                f"[{index}/{total}] {status} chunks={book_report.chunk_count} "
                f"inline_toc={inline_toc} weak_rungs={weak}"
            ),
        )
    return CorpusReport(tuple(reports))


def aggregate_category_audit(
    corpus_report: CorpusReport,
    *,
    books_per_category: int = 1,
) -> CategoryAuditReport:
    grouped: dict[int, list[BookReport]] = defaultdict(list)
    for book in corpus_report.books:
        if book.category_id is not None:
            grouped[book.category_id].append(book)

    rows: list[CategoryAuditRow] = []
    for category_id in sorted(grouped):
        books = grouped[category_id]
        boundary_counts: dict[str, int] = defaultdict(int)
        confidence_counts: dict[str, int] = defaultdict(int)
        ambiguous = 0
        for book in books:
            for source, count in book.boundary_source_counts.items():
                boundary_counts[source] += count
            for confidence, count in book.confidence_counts.items():
                confidence_counts[confidence] += count
            ambiguous += book.ambiguous_boundaries
        rows.append(
            CategoryAuditRow(
                category_id=category_id,
                book_count=len(books),
                boundary_source_counts=dict(boundary_counts),
                confidence_counts=dict(confidence_counts),
                ambiguous_boundaries=ambiguous,
                low_confidence_boundaries=confidence_counts.get("low", 0),
            )
        )
    return CategoryAuditReport(tuple(rows), books_per_category=books_per_category)


def recommend_category(row: CategoryAuditRow) -> str:
    total = row.total_boundaries
    if total == 0:
        return "No boundaries detected in the sample; verify TOC/page markup before scale ingest."
    weak = (
        row.boundary_source_counts.get(BoundarySource.PARAGRAPH_FALLBACK.value, 0)
        + row.boundary_source_counts.get(BoundarySource.AMBIGUOUS_TOC_PAGE.value, 0)
        + row.boundary_source_counts.get(BoundarySource.TOC_PAGE_FALLBACK.value, 0)
    )
    weak_pct = 100.0 * weak / total
    inline_toc_pct = row.boundary_pct(BoundarySource.INLINE_TOC.value)
    if weak_pct >= 50:
        return (
            "Heavy reliance on weak ladder rungs; evaluate a category-specific chunker "
            "before scale ingest (do not build one until this audit confirms it)."
        )
    if inline_toc_pct >= 50:
        return "Generic structural chunker is well served (majority inline_toc boundaries)."
    return "Mixed boundary quality; monitor during pilot ingest before changing strategy."


def validate_category_audit(
    corpus_root: Path,
    *,
    books_per_category: int = 1,
    book_id: int | None = None,
    category_id: int | None = None,
    policy: SizePolicy = DEFAULT_POLICY,
    min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS,
    progress: ProgressFn | None = None,
) -> tuple[CorpusReport, CategoryAuditReport]:
    _log_progress(progress, f"corpus root: {corpus_root}")
    corpus_report = validate_corpus(
        corpus_root,
        limit=None,
        book_id=book_id,
        category_id=category_id,
        stratified=True,
        books_per_category=books_per_category,
        policy=policy,
        min_content_tokens=min_content_tokens,
        progress=progress,
    )
    _log_progress(progress, "aggregating per-category boundary statistics...")
    audit_report = aggregate_category_audit(corpus_report, books_per_category=books_per_category)
    _log_progress(
        progress,
        f"audit complete: {audit_report.category_count} categor"
        f"{'y' if audit_report.category_count == 1 else 'ies'} in report",
    )
    return corpus_report, audit_report


def format_report(report: BookReport | CorpusReport) -> str:
    if isinstance(report, BookReport):
        return _format_book(report)

    lines = [
        f"structural validation: {len(report.books)} book(s), "
        f"{report.hard_violation_count} hard violation(s)",
        "",
    ]
    for book in report.books:
        lines.append(_format_book(book))
        lines.append("")
    if report.ok:
        lines.append("PASS: zero hard violations")
    elif not report.books:
        lines.append("FAIL: no books validated")
    else:
        lines.append(f"FAIL: {report.hard_violation_count} hard violation(s)")
    return "\n".join(lines).rstrip() + "\n"


def format_category_audit_report(report: CategoryAuditReport) -> str:
    lines = [
        "# Structural chunking per-category audit",
        "",
        (
            f"Sample: up to {report.books_per_category} book(s) per category "
            f"({report.category_count} categor{'y' if report.category_count == 1 else 'ies'})."
        ),
        "",
        (
            "| category_id | books | boundaries | inline_toc % | paragraph_fallback % | "
            "ambiguous | low_conf | recommendation |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.category_id),
                    str(row.book_count),
                    str(row.total_boundaries),
                    f"{row.boundary_pct(BoundarySource.INLINE_TOC.value):.1f}",
                    f"{row.boundary_pct(BoundarySource.PARAGRAPH_FALLBACK.value):.1f}",
                    str(row.ambiguous_boundaries),
                    str(row.low_confidence_boundaries),
                    recommend_category(row),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary sources by category",
            "",
        ]
    )
    source_order = [source.value for source in BoundarySource]
    for row in report.rows:
        lines.append(f"### category_id={row.category_id}")
        for source in source_order:
            count = row.boundary_source_counts.get(source, 0)
            if count:
                lines.append(f"- {source}: {count}")
        lines.append("")
    lines.extend(
        [
            "## Recommendations",
            "",
            "The general-QA module uses one generic structural path for all genres today. "
            "Only pursue a category-specific chunker when this audit shows sustained "
            "reliance on weak ladder rungs; do not build one speculatively.",
            "",
        ]
    )
    for row in report.rows:
        lines.append(f"- **category {row.category_id}**: {recommend_category(row)}")
    lines.append("")
    return "\n".join(lines)


def _format_book(report: BookReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"[{status}] book_id={report.book_id} chunks={report.chunk_count} path={report.book_dir}",
    ]
    if report.confidence_counts:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(report.confidence_counts.items()))
        lines.append(f"  confidence: {counts}")
    if report.boundary_source_counts:
        sources = ", ".join(f"{k}={v}" for k, v in sorted(report.boundary_source_counts.items()))
        lines.append(f"  boundary_sources: {sources}")
    if report.ambiguous_boundaries:
        lines.append(f"  ambiguous_toc_page: {report.ambiguous_boundaries}")
    if report.heading_recovery_candidates:
        lines.append(f"  heading_recovery_candidates: {report.heading_recovery_candidates}")
    for finding in report.findings:
        page = f" page={finding.page_id}" if finding.page_id is not None else ""
        lines.append(f"  {finding.severity.value.upper()} {finding.check}{page}: {finding.message}")
    if report.ok and not report.findings:
        lines.append("  (no findings)")
    return "\n".join(lines)
