"""Parser for inline title markup embedded in page bodies.

Bodies carry ``<span data-type='title' id=toc-N>heading</span>`` markers. This inline layer is
richer than ``toc.jsonl`` (it can mark sub-page headings the exported TOC omits), so it is the
strongest boundary source. The ``id`` is ``toc-<N>`` where ``N`` is the ``shamela_title_id`` (NOT
the exported global ``title_id``). Attribute quoting (none/single/double) and order both vary, so
the parser is tolerant of all three.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPAN = re.compile(r"<span\b([^>]*)>(.*?)</span>", re.DOTALL)
_TYPE = re.compile(r"data-type=(['\"]?)title\1")
_ID = re.compile(r"id=(['\"]?)toc-(\d+)\1")


@dataclass(frozen=True)
class TitleSpan:
    shamela_title_id: int  # from `toc-N`; maps to toc.jsonl.shamela_title_id
    title_text: str
    start: int  # offset of '<span' in the body
    end: int  # offset just past '</span>'


def parse_title_spans(body: str) -> list[TitleSpan]:
    spans: list[TitleSpan] = []
    for match in _SPAN.finditer(body):
        attrs = match.group(1)
        if _TYPE.search(attrs) is None:
            continue
        id_match = _ID.search(attrs)
        if id_match is None:
            continue
        spans.append(
            TitleSpan(
                shamela_title_id=int(id_match.group(2)),
                title_text=match.group(2).strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    return spans
