"""Classify a section's body as navigational or content.

Navigational nodes (volume labels, alphabet ranges like ``حرف الألف``, empty ``باب`` headings,
dividers) carry a heading but little or no prose. They are kept as parent context for their
children but must NOT become searchable embedding chunks. Classification strips the inline title
markup and measures what prose remains against a (configurable) minimum token count.
"""

from __future__ import annotations

import re
from enum import StrEnum

from shamela_rag.chunking.tokens import count_tokens
from shamela_rag.text.normalization import normalize_for_display

_TITLE_SPAN = re.compile(r"<span\b[^>]*\bdata-type=(['\"]?)title\1[^>]*>.*?</span>", re.DOTALL)

DEFAULT_MIN_CONTENT_TOKENS = 5


class NodeKind(StrEnum):
    CONTENT = "content"
    NAVIGATIONAL = "navigational"


def strip_title_markup(body: str) -> str:
    return _TITLE_SPAN.sub(" ", body)


def substantive_text(body: str) -> str:
    return normalize_for_display(strip_title_markup(body)).strip()


def classify_body(body: str, *, min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS) -> NodeKind:
    if count_tokens(substantive_text(body)) < min_content_tokens:
        return NodeKind.NAVIGATIONAL
    return NodeKind.CONTENT


def is_navigational(body: str, *, min_content_tokens: int = DEFAULT_MIN_CONTENT_TOKENS) -> bool:
    return classify_body(body, min_content_tokens=min_content_tokens) is NodeKind.NAVIGATIONAL
