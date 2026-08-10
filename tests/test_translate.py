from __future__ import annotations

import pytest

from shamela_rag.retrieval import (
    InMemoryTranslator,
    PreparedQuery,
    QueryLanguage,
    detect_query_language,
    prepare_retrieval_query,
)
from shamela_rag.retrieval.translate import Translator


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ما قول ابن تيمية في الصوفية؟", QueryLanguage.ARABIC),
        # Diacritized classical Arabic still counts as Arabic script.
        ("مَا قَوْلُ ابْنِ تَيْمِيَّةَ؟", QueryLanguage.ARABIC),
        ("What did Ibn Taymiyyah say about the Sufis?", QueryLanguage.ENGLISH),
        ("what is salah?", QueryLanguage.ENGLISH),
        ("WHAT IS SALAH?", QueryLanguage.ENGLISH),
        # Mixed: Latin shell + Arabic term must still be treated as needing translation.
        ("What is الصلاة?", QueryLanguage.ENGLISH),
        ("Ibn Taymiyyah and الصوفية", QueryLanguage.ENGLISH),
        ("Explain التيمم briefly", QueryLanguage.ENGLISH),
        # Transliteration only (no Arabic script) → English path.
        ("ma qawl ibn taymiyyah fi al-sufiyya", QueryLanguage.ENGLISH),
        # French / other Latin still goes through the translate path (same "non-Arabic Latin").
        ("Quelle est la position d'Ibn Taymiyya?", QueryLanguage.ENGLISH),
        ("", QueryLanguage.OTHER),
        ("   ", QueryLanguage.OTHER),
        ("12345", QueryLanguage.OTHER),
        ("١٢٣٤٥", QueryLanguage.OTHER),  # Eastern Arabic-Indic digits live in Arabic block
        ("???", QueryLanguage.OTHER),
        ("—…", QueryLanguage.OTHER),
        ("\n\t", QueryLanguage.OTHER),
    ],
)
def test_detect_query_language(text: str, expected: QueryLanguage) -> None:
    assert detect_query_language(text) == expected


def test_detect_multiline_english() -> None:
    text = "What did he say\nabout combining prayers while traveling?"
    assert detect_query_language(text) == QueryLanguage.ENGLISH


def test_detect_multiline_arabic() -> None:
    text = "ما قوله\nفي جمع الصلاة في السفر؟"
    assert detect_query_language(text) == QueryLanguage.ARABIC


def test_in_memory_translator_uses_mapping() -> None:
    translator = InMemoryTranslator(
        {
            "What did Ibn Taymiyyah say about the Sufis?": "ما قول ابن تيمية في الصوفية؟",
        }
    )
    out = translator.translate(
        "What did Ibn Taymiyyah say about the Sufis?",
        source=QueryLanguage.ENGLISH,
        target=QueryLanguage.ARABIC,
    )
    assert out == "ما قول ابن تيمية في الصوفية؟"
    assert len(translator.calls) == 1
    assert translator.calls[0].source == QueryLanguage.ENGLISH
    assert translator.calls[0].target == QueryLanguage.ARABIC


def test_in_memory_translator_falls_back_to_prefix() -> None:
    translator = InMemoryTranslator(prefix="AR:")
    assert (
        translator.translate(
            "hello",
            source=QueryLanguage.ENGLISH,
            target=QueryLanguage.ARABIC,
        )
        == "AR:hello"
    )


def test_in_memory_translator_same_language_is_identity() -> None:
    translator = InMemoryTranslator({"a": "b"})
    assert (
        translator.translate("a", source=QueryLanguage.ARABIC, target=QueryLanguage.ARABIC) == "a"
    )


def test_english_question_is_translated_for_retrieval() -> None:
    english = "What did Ibn Taymiyyah say about the Sufis?"
    arabic = "ما قول ابن تيمية في الصوفية؟"
    prepared = prepare_retrieval_query(english, InMemoryTranslator({english: arabic}))

    assert prepared == PreparedQuery(
        original=english,
        retrieval_text=arabic,
        source_language=QueryLanguage.ENGLISH,
        was_translated=True,
    )
    assert prepared.original == english
    assert prepared.retrieval_text == arabic


def test_arabic_question_passthrough_does_not_call_translator() -> None:
    arabic = "ما قول ابن تيمية في الصوفية؟"
    translator = InMemoryTranslator({arabic: "SHOULD_NOT_APPEAR"})
    prepared = prepare_retrieval_query(arabic, translator)

    assert translator.calls == []
    assert prepared.was_translated is False
    assert prepared.source_language == QueryLanguage.ARABIC
    assert prepared.original == arabic
    assert prepared.retrieval_text == arabic


def test_mixed_english_arabic_term_is_translated() -> None:
    question = "What is الصلاة?"
    translated = "ما هي الصلاة؟"
    translator = InMemoryTranslator({question: translated})
    prepared = prepare_retrieval_query(question, translator)

    assert prepared.was_translated is True
    assert prepared.source_language == QueryLanguage.ENGLISH
    assert prepared.retrieval_text == translated
    assert len(translator.calls) == 1


def test_prepare_strips_surrounding_whitespace() -> None:
    prepared = prepare_retrieval_query(
        "  hello world  ",
        InMemoryTranslator({"hello world": "مرحبا"}),
    )
    assert prepared.original == "hello world"
    assert prepared.retrieval_text == "مرحبا"
    assert prepared.was_translated is True


def test_prepare_preserves_internal_whitespace_and_newlines() -> None:
    english = "line one\nline two"
    arabic = "سطر واحد\nسطر اثنان"
    prepared = prepare_retrieval_query(english, InMemoryTranslator({english: arabic}))
    assert prepared.original == english
    assert prepared.retrieval_text == arabic


def test_other_language_passthrough_does_not_call_translator() -> None:
    translator = InMemoryTranslator(prefix="X")
    prepared = prepare_retrieval_query("12345", translator)
    assert translator.calls == []
    assert prepared.source_language == QueryLanguage.OTHER
    assert prepared.was_translated is False
    assert prepared.retrieval_text == "12345"


def test_long_complex_english_question_is_translated() -> None:
    english = (
        "According to Ibn Taymiyyah, how should one weigh the opinions of earlier scholars "
        "when they disagree about combining prayers while traveling, and which primary "
        "passages does he cite?"
    )
    arabic = "نص عربي طويل للتحقق"
    translator = InMemoryTranslator({english: arabic})
    prepared = prepare_retrieval_query(english, translator)
    assert prepared.was_translated is True
    assert prepared.retrieval_text == arabic
    assert prepared.original == english


class _EmptyTranslator(Translator):
    def translate(
        self,
        text: str,
        *,
        source: QueryLanguage,
        target: QueryLanguage,
    ) -> str:
        return "   "


def test_empty_translator_result_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_retrieval_query("hello", _EmptyTranslator())


def test_prepared_query_is_immutable() -> None:
    prepared = prepare_retrieval_query("hello", InMemoryTranslator({"hello": "مرحبا"}))
    with pytest.raises(AttributeError):
        prepared.was_translated = False  # type: ignore[misc]
