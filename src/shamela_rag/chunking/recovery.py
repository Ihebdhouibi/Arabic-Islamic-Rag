"""Optional recovery of heading-like text that is absent from both the TOC and inline markup.

Some pages contain a visible heading (e.g. an unmarked biography title next to a marked one) that
neither ``toc.jsonl`` nor a ``<span data-type=title>`` records. We surface these as *candidates*
with a confidence and pattern, but we never split on them here: they must be measured for precision
first (M6). Lines already covered by an inline title span, or matching a known TOC title, are
skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shamela_rag.chunking.boundaries import Confidence
from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.chunking.tokens import count_tokens

_LINE = re.compile(r"[^\r\n]+")
# Leading number (ASCII or Arabic-Indic digits) then a dash: "١ - آبي اللحم" / "12 - Name".
_NUMBERED_ENTRY = re.compile(r"^\s*[0-9\u0660-\u0669]+\s*[-\u2013\u2014]\s*\S")
_SENTENCE_END = ".!?؟؛,،"

DEFAULT_MAX_HEADING_TOKENS = 8


@dataclass(frozen=True)
class HeadingCandidate:
    offset: int
    text: str
    confidence: Confidence
    pattern: str


def _classify_line(line: str, max_heading_tokens: int) -> tuple[str, Confidence] | None:
    if _NUMBERED_ENTRY.match(line):
        return "numbered_entry", Confidence.MEDIUM
    if count_tokens(line) <= max_heading_tokens and line[-1] not in _SENTENCE_END:
        return "short_line", Confidence.LOW
    return None


def recover_heading_candidates(
    body: str,
    *,
    toc_titles: frozenset[str] = frozenset(),
    max_heading_tokens: int = DEFAULT_MAX_HEADING_TOKENS,
) -> list[HeadingCandidate]:
    span_ranges = [(span.start, span.end) for span in parse_title_spans(body)]
    candidates: list[HeadingCandidate] = []
    for match in _LINE.finditer(body):
        start = match.start()
        line = match.group().strip()
        if not line:
            continue
        if any(lo <= start < hi for lo, hi in span_ranges):
            continue  # already marked inline
        if line in toc_titles:
            continue  # already in the TOC
        classified = _classify_line(line, max_heading_tokens)
        if classified is not None:
            pattern, confidence = classified
            candidates.append(HeadingCandidate(start, line, confidence, pattern))
    return candidates
