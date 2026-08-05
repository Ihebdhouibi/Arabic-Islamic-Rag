"""Deterministic page ordering and a source-offset model.

Pages are ordered by the internal ``page_id`` (monotonic per book). ``sequence_num`` must NOT be
used as an ordering key: it is not unique within a book (e.g. it can restart per part), and printed
``page_num`` restarts across volumes. Bodies are concatenated into one per-book stream so later
chunking can map a chunk's character range back to the page(s) it came from; each page's body
appears verbatim at its span, and inter-page separators are the only characters not owned by a page.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from shamela_rag.data import models
from shamela_rag.data.models import Page


@dataclass(frozen=True)
class PageSpan:
    page: Page
    start: int  # inclusive char offset into SourceStream.text
    end: int  # exclusive


@dataclass(frozen=True)
class SourceStream:
    text: str
    spans: tuple[PageSpan, ...]

    def page_at(self, offset: int) -> PageSpan | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span
        return None


def order_pages(pages: Iterable[Page]) -> list[Page]:
    return sorted(pages, key=lambda p: p.page_id)


def build_source_stream(pages: Iterable[Page], *, separator: str = "\n") -> SourceStream:
    ordered = order_pages(pages)
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for index, page in enumerate(ordered):
        if index > 0:
            parts.append(separator)
            cursor += len(separator)
        start = cursor
        parts.append(page.body)
        cursor += len(page.body)
        spans.append(PageSpan(page=page, start=start, end=cursor))
    return SourceStream(text="".join(parts), spans=tuple(spans))


def load_source_stream(book_dir: Path, *, separator: str = "\n") -> SourceStream:
    return build_source_stream(models.load_pages(book_dir), separator=separator)
