"""Per-page boundary detection via a fallback ladder.

For each page we pick the strongest available boundary evidence and record where it came from plus a
confidence, so uncertain boundaries can be inspected and improved without re-extracting the source.
Priority (strongest first):

1. ``inline_toc``          -- inline ``<span data-type=title id=toc-N>`` (maps to shamela_title_id)
2. ``inline_title``        -- inline title span with no ``toc-N`` id (structural, no TOC identity)
3. ``recovered_title``     -- a TOC entry whose title text is found verbatim in the page body
4. ``toc_page_fallback``   -- a single unmatched TOC entry: fall back to page start (low confidence)
5. ``ambiguous_toc_page``  -- several unmatched TOC entries: do NOT fabricate offsets
6. ``paragraph_fallback``  -- no usable TOC: split on paragraph breaks
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.data.models import TocEntry


class BoundarySource(StrEnum):
    INLINE_TOC = "inline_toc"
    INLINE_TITLE = "inline_title"
    RECOVERED_TITLE = "recovered_title"
    TOC_PAGE_FALLBACK = "toc_page_fallback"
    AMBIGUOUS_TOC_PAGE = "ambiguous_toc_page"
    PARAGRAPH_FALLBACK = "paragraph_fallback"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Boundary:
    offset: int
    source: BoundarySource
    confidence: Confidence
    shamela_title_id: int | None = None
    title_text: str | None = None


_TITLE_OPEN = re.compile(r"<span\b[^>]*\bdata-type=(['\"]?)title\1[^>]*>")
_HAS_TOC_ID = re.compile(r"\bid=(['\"]?)toc-\d+\1")
_PARAGRAPH_BREAK = re.compile(r"(?:\r\n|\r|\n){2,}")


def _inline_title_only_starts(body: str) -> list[int]:
    return [m.start() for m in _TITLE_OPEN.finditer(body) if _HAS_TOC_ID.search(m.group(0)) is None]


def _paragraph_starts(body: str) -> list[int]:
    starts = [0]
    starts.extend(m.end() for m in _PARAGRAPH_BREAK.finditer(body))
    return starts


def detect_page_boundaries(body: str, toc_entries: Sequence[TocEntry]) -> list[Boundary]:
    """Detect boundaries for one page. ``toc_entries`` are those anchored to this page."""
    boundaries: list[Boundary] = []

    for span in parse_title_spans(body):
        boundaries.append(
            Boundary(
                span.start,
                BoundarySource.INLINE_TOC,
                Confidence.HIGH,
                span.shamela_title_id,
                span.title_text,
            )
        )
    for start in _inline_title_only_starts(body):
        boundaries.append(Boundary(start, BoundarySource.INLINE_TITLE, Confidence.MEDIUM))
    if boundaries:
        return sorted(boundaries, key=lambda b: b.offset)

    if toc_entries:
        unmatched: list[TocEntry] = []
        for entry in toc_entries:
            idx = body.find(entry.title_text) if entry.title_text else -1
            if idx >= 0:
                boundaries.append(
                    Boundary(
                        idx,
                        BoundarySource.RECOVERED_TITLE,
                        Confidence.MEDIUM,
                        entry.shamela_title_id,
                        entry.title_text,
                    )
                )
            else:
                unmatched.append(entry)
        if len(unmatched) == 1:
            boundaries.append(
                Boundary(
                    0,
                    BoundarySource.TOC_PAGE_FALLBACK,
                    Confidence.LOW,
                    unmatched[0].shamela_title_id,
                    unmatched[0].title_text,
                )
            )
        elif len(unmatched) >= 2:
            boundaries.append(Boundary(0, BoundarySource.AMBIGUOUS_TOC_PAGE, Confidence.LOW))
        return sorted(boundaries, key=lambda b: b.offset)

    return [
        Boundary(start, BoundarySource.PARAGRAPH_FALLBACK, Confidence.LOW)
        for start in _paragraph_starts(body)
    ]


def confidence_counts(boundaries: Iterable[Boundary]) -> dict[Confidence, int]:
    counts = dict.fromkeys(Confidence, 0)
    for boundary in boundaries:
        counts[boundary.confidence] += 1
    return counts
