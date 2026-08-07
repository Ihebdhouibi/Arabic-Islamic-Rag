"""Per-book chunking orchestrator: compose the M2 stages into ``chunk_book``.

Pipeline per page: order -> content-role split -> boundary ladder -> structural trail ->
navigational skip -> size split -> context header. Every emitted chunk's ``source_text`` is an exact
verbatim slice of the page ``body`` or ``footnotes`` (never mutated); ``retrieval_text`` holds the
normalized form. Navigational, heading-only segments are not emitted as chunks but still update the
trail their children inherit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shamela_rag.chunking.boundaries import Boundary, detect_page_boundaries
from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.context_header import build_context_header, context_from
from shamela_rag.chunking.navigation import DEFAULT_MIN_CONTENT_TOKENS, is_navigational
from shamela_rag.chunking.sections import build_sections
from shamela_rag.chunking.sizing import DEFAULT_POLICY, SizePolicy, split_section_offsets
from shamela_rag.chunking.tokens import count_tokens
from shamela_rag.data.models import Book, Page, TocEntry, load_book, load_pages, load_toc
from shamela_rag.data.ordering import order_pages
from shamela_rag.text.normalization import normalize_for_index


@dataclass(frozen=True)
class BookChunk:
    book_id: int
    content_role: ContentRole
    source_text: str  # verbatim slice of the page body/footnotes
    retrieval_text: str  # normalized for indexing
    context_header: str
    trail: tuple[str, ...]
    boundary_source: str | None
    confidence: str | None
    page_id: int
    start_offset: int  # offset within the page body/footnotes
    end_offset: int
    token_count: int


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    boundary: Boundary | None


def _segments(body: str, boundaries: Sequence[Boundary]) -> list[_Segment]:
    ordered = sorted(boundaries, key=lambda b: b.offset)
    points: list[tuple[int, Boundary | None]] = []
    if not ordered or ordered[0].offset > 0:
        points.append((0, None))
    points.extend((b.offset, b) for b in ordered)

    segments: list[_Segment] = []
    for index, (start, boundary) in enumerate(points):
        end = points[index + 1][0] if index + 1 < len(points) else len(body)
        if end > start:
            segments.append(_Segment(start, end, boundary))
    return segments


def _book_root(book: Book) -> tuple[str, ...]:
    return (book.title_ar,) if book.title_ar else ()


def _make_chunk(
    book: Book,
    role: ContentRole,
    page: Page,
    start: int,
    end: int,
    source_text: str,
    trail: tuple[str, ...],
    boundary: Boundary | None,
) -> BookChunk:
    header = build_context_header(context_from(book, trail, role))
    return BookChunk(
        book_id=book.book_id,
        content_role=role,
        source_text=source_text,
        retrieval_text=normalize_for_index(source_text),
        context_header=header,
        trail=trail,
        boundary_source=boundary.source.value if boundary is not None else None,
        confidence=boundary.confidence.value if boundary is not None else None,
        page_id=page.page_id,
        start_offset=start,
        end_offset=end,
        token_count=count_tokens(source_text),
    )


def chunk_book(
    book_dir: Path,
    *,
    policy: SizePolicy = DEFAULT_POLICY,
    min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS,
) -> list[BookChunk]:
    directory = book_dir
    book = load_book(directory)
    toc: list[TocEntry] = list(load_toc(directory))
    trail_by_title_id = {s.shamela_title_id: s.trail for s in build_sections(toc)}

    chunks: list[BookChunk] = []
    trail: tuple[str, ...] = _book_root(book)

    for page in order_pages(load_pages(directory)):
        toc_on_page = [entry for entry in toc if entry.page_id == page.page_id]
        for segment in _segments(page.body, detect_page_boundaries(page.body, toc_on_page)):
            boundary = segment.boundary
            if boundary is not None and boundary.shamela_title_id in trail_by_title_id:
                trail = trail_by_title_id[boundary.shamela_title_id]
            elif boundary is not None and boundary.title_text:
                trail = (*_book_root(book), boundary.title_text)

            segment_text = page.body[segment.start : segment.end]
            if is_navigational(segment_text, min_content_tokens=min_content_tokens):
                continue
            for local_start, local_end in split_section_offsets(segment_text, policy):
                source = segment_text[local_start:local_end]
                if not source.strip():
                    continue
                chunks.append(
                    _make_chunk(
                        book,
                        ContentRole.BODY,
                        page,
                        segment.start + local_start,
                        segment.start + local_end,
                        source,
                        trail,
                        boundary,
                    )
                )

        if page.footnotes and page.footnotes.strip():
            for local_start, local_end in split_section_offsets(page.footnotes, policy):
                source = page.footnotes[local_start:local_end]
                if not source.strip():
                    continue
                chunks.append(
                    _make_chunk(
                        book,
                        ContentRole.FOOTNOTE,
                        page,
                        local_start,
                        local_end,
                        source,
                        trail,
                        None,
                    )
                )

    return chunks
