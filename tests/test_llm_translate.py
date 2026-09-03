from __future__ import annotations

import io
import json
from typing import Any

import pytest

from shamela_rag.config import get_settings
from shamela_rag.factory import build_translator
from shamela_rag.retrieval.llm_translate import OpenAICompatibleTranslator
from shamela_rag.retrieval.translate import QueryLanguage, prepare_retrieval_query


def _fake_opener(payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode("utf-8")

    class _Response(io.BytesIO):
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def opener(request: Any, timeout: float = 30.0) -> _Response:
        del timeout
        return _Response(body)

    return opener


def test_openai_compatible_translator_returns_arabic() -> None:
    arabic = "ما قول ابن تيمية في الصوفية؟"
    translator = OpenAICompatibleTranslator(
        "test-model",
        api_key="secret",
        opener=_fake_opener(
            {
                "choices": [{"message": {"content": arabic}}],
            }
        ),
    )
    out = translator.translate(
        "What did Ibn Taymiyyah say about the Sufis?",
        source=QueryLanguage.ENGLISH,
        target=QueryLanguage.ARABIC,
    )
    assert out == arabic


def test_openai_compatible_translator_rejects_unsupported_pair() -> None:
    translator = OpenAICompatibleTranslator(
        "test-model",
        api_key="secret",
        opener=_fake_opener({"choices": [{"message": {"content": "x"}}]}),
    )
    with pytest.raises(ValueError, match="EN→AR"):
        translator.translate(
            "text",
            source=QueryLanguage.ARABIC,
            target=QueryLanguage.ENGLISH,
        )


def test_openai_compatible_translator_raises_on_empty_response() -> None:
    translator = OpenAICompatibleTranslator(
        "test-model",
        api_key="secret",
        opener=_fake_opener({"choices": [{"message": {"content": "   "}}]}),
    )
    with pytest.raises(ValueError, match="empty"):
        translator.translate(
            "hello",
            source=QueryLanguage.ENGLISH,
            target=QueryLanguage.ARABIC,
        )


def test_prepare_retrieval_query_with_llm_translator() -> None:
    arabic = "طلب العلم فريضة"
    translator = OpenAICompatibleTranslator(
        "test-model",
        api_key="secret",
        opener=_fake_opener({"choices": [{"message": {"content": arabic}}]}),
    )
    prepared = prepare_retrieval_query("Seeking knowledge is an obligation", translator)
    assert prepared.was_translated is True
    assert prepared.retrieval_text == arabic


def test_build_translator_defaults_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHAMELA_TRANSLATOR_BACKEND", raising=False)
    get_settings.cache_clear()
    try:
        from shamela_rag.retrieval.translate import InMemoryTranslator

        assert isinstance(build_translator(), InMemoryTranslator)
    finally:
        get_settings.cache_clear()


def test_build_translator_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAMELA_TRANSLATOR_BACKEND", "openai_compatible")
    monkeypatch.setenv("SHAMELA_LLM_API_MODEL", "remote-model")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="SHAMELA_LLM_API_KEY"):
            build_translator()
    finally:
        get_settings.cache_clear()


def test_build_translator_openai_uses_translator_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAMELA_TRANSLATOR_BACKEND", "openai_compatible")
    monkeypatch.setenv("SHAMELA_LLM_API_KEY", "secret")
    monkeypatch.setenv("SHAMELA_TRANSLATOR_API_MODEL", "translate-model")
    get_settings.cache_clear()
    try:
        translator = build_translator()
        assert isinstance(translator, OpenAICompatibleTranslator)
        assert translator._model == "translate-model"
    finally:
        get_settings.cache_clear()
