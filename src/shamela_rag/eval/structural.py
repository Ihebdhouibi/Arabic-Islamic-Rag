"""Structural validation harness for the chunking pipeline (M6-06).

Runs ``chunk_book`` over books and checks hard invariants: verbatim offsets, source coverage,
determinism, inline ``toc-N`` boundaries, section integrity, within-section overlap, and footnote
roles. Ambiguous / low-confidence boundaries are reported as soft findings and do not fail the gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from shamela_rag.chunking.boundaries import Boundary, BoundarySource, detect_page_boundaries
from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.navigation import DEFAULT_MIN_CONTENT_TOKENS, is_navigational
from shamela_rag.chunking.orchestrator import BookChunk, ChunkingResult, chunk_book
from shamela_rag.chunking.sizing import DEFAULT_POLICY, SizePolicy
from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.data.discovery import BookLocation, iter_valid_books
from shamela_rag.data.models import Page, TocEntry, load_book, load_pages, load_toc
from shamela_rag.data.ordering import order_pages


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
    findings: list[Finding] = field(default_factory=list)
    confidence_counts: dict[str, int] = field(default_factory=dict)
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
        findings=findings,
        confidence_counts=dict(result.stats.confidence_counts),
        ambiguous_boundaries=ambiguous,
        heading_recovery_candidates=result.stats.heading_recovery_candidates,
    )


def validate_corpus(
    corpus_root: Path,
    *,
    limit: int | None = 20,
    book_id: int | None = None,
    category_id: int | None = None,
    policy: SizePolicy = DEFAULT_POLICY,
    min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS,
) -> CorpusReport:
    locations: list[BookLocation] = list(iter_valid_books(corpus_root))
    if book_id is not None:
        locations = [loc for loc in locations if loc.book_id == book_id]
    if category_id is not None:
        locations = [loc for loc in locations if loc.category_id == category_id]
    if limit is not None:
        locations = locations[:limit]

    reports = [
        validate_book(loc.book_dir, policy=policy, min_content_tokens=min_content_tokens)
        for loc in locations
    ]
    return CorpusReport(tuple(reports))


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


def _format_book(report: BookReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    lines = [
        f"[{status}] book_id={report.book_id} chunks={report.chunk_count} path={report.book_dir}",
    ]
    if report.confidence_counts:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(report.confidence_counts.items()))
        lines.append(f"  confidence: {counts}")
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
