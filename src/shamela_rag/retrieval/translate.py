"""Query language detection and EN→AR translation for the retrieval path.

Production retrieval runs against an Arabic corpus, so English questions are translated
before dense/sparse search. The original wording is kept for display; the Arabic form is
what retrieval (and eval parity) use. Arabic questions pass through unchanged.

Detection is intentionally conservative for mixed queries: any Latin letters mean we treat
the question as needing EN→AR translation (so ``What is الصلاة؟`` is translated, not left
as an English shell around an Arabic term). Pure Arabic script passes through.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class QueryLanguage(StrEnum):
    ARABIC = "ar"
    ENGLISH = "en"
    OTHER = "other"


# Arabic script blocks commonly seen in classical + modern Arabic text.
_ARABIC_CHARS = frozenset(
    chr(c)
    for c in (
        *range(0x0600, 0x06FF + 1),  # Arabic
        *range(0x0750, 0x077F + 1),  # Arabic Supplement
        *range(0x08A0, 0x08FF + 1),  # Arabic Extended-A
        *range(0xFB50, 0xFDFF + 1),  # Arabic Presentation Forms-A
        *range(0xFE70, 0xFEFF + 1),  # Arabic Presentation Forms-B
    )
)


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """Result of preparing a user question for retrieval.

    ``original`` is what the UI/eval may show; ``retrieval_text`` is the Arabic string
    actually sent to dense/sparse search. ``was_translated`` is True only when an EN→AR
    translation step ran.
    """

    original: str
    retrieval_text: str
    source_language: QueryLanguage
    was_translated: bool


class Translator(ABC):
    """Pluggable EN→AR (or other) translator for the production retrieval path."""

    @abstractmethod
    def translate(
        self,
        text: str,
        *,
        source: QueryLanguage,
        target: QueryLanguage,
    ) -> str:
        """Return ``text`` translated from ``source`` into ``target``."""


@dataclass
class _TranslateCall:
    text: str
    source: QueryLanguage
    target: QueryLanguage


class InMemoryTranslator(Translator):
    """Deterministic, offline translator (dict mapping; no network / model weights).

    Looks up ``text`` in ``mapping``; if missing, returns ``prefix + text`` so tests can
    assert that translation was invoked without shipping a real MT backend. Records every
    ``translate`` call in ``calls`` for passthrough / spy assertions.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        *,
        prefix: str = "[ar] ",
    ) -> None:
        self._mapping = dict(mapping) if mapping is not None else {}
        self._prefix = prefix
        self.calls: list[_TranslateCall] = []

    def translate(
        self,
        text: str,
        *,
        source: QueryLanguage,
        target: QueryLanguage,
    ) -> str:
        self.calls.append(_TranslateCall(text=text, source=source, target=target))
        if source == target:
            return text
        if text in self._mapping:
            return self._mapping[text]
        return f"{self._prefix}{text}"


def detect_query_language(text: str) -> QueryLanguage:
    """Classify a question as Arabic, English, or other via script heuristics.

    - Pure Arabic script → ``ARABIC`` (passthrough).
    - Any Latin letters → ``ENGLISH`` (needs translation), including mixed
      English+Arabic-term questions.
    - Empty / digits / punctuation-only → ``OTHER``.
    """
    stripped = text.strip()
    if not stripped:
        return QueryLanguage.OTHER

    arabic = 0
    latin = 0
    for ch in stripped:
        # Letters only: Eastern digits / tashkeel / punctuation in the Arabic block
        # must not alone force ARABIC (or they'd mis-label "١٢٣" as Arabic prose).
        if ch in _ARABIC_CHARS and ch.isalpha():
            arabic += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1

    if latin > 0:
        return QueryLanguage.ENGLISH
    if arabic > 0:
        return QueryLanguage.ARABIC
    return QueryLanguage.OTHER


def prepare_retrieval_query(question: str, translator: Translator) -> PreparedQuery:
    """Prepare a user question for the Arabic retrieval path.

    - Arabic → passthrough (``retrieval_text is original``, not translated).
    - English (including mixed Latin+Arabic) → translate to Arabic; keep ``original``.
    - Other → passthrough (do not invent a translation).

    Raises:
        ValueError: if the translator returns an empty/whitespace-only string.
    """
    original = question.strip()
    language = detect_query_language(original)

    if language == QueryLanguage.ENGLISH:
        translated = translator.translate(
            original,
            source=QueryLanguage.ENGLISH,
            target=QueryLanguage.ARABIC,
        )
        if not translated.strip():
            raise ValueError("translator returned empty text for retrieval")
        return PreparedQuery(
            original=original,
            retrieval_text=translated,
            source_language=language,
            was_translated=True,
        )

    return PreparedQuery(
        original=original,
        retrieval_text=original,
        source_language=language,
        was_translated=False,
    )
