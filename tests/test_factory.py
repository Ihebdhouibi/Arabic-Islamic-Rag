from __future__ import annotations

import json
from pathlib import Path

import pytest

from shamela_rag import cli
from shamela_rag.config import get_settings
from shamela_rag.data.root_dictionary import load_root_dictionary
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.embeddings.root_field import RootExpansionEncoder
from shamela_rag.generation.answer import Answer, Citation
from shamela_rag.generation.provider import InMemoryGenerationProvider
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.rerank import LexicalOverlapReranker
from shamela_rag.retrieval.translate import InMemoryTranslator

_ANSWER = Answer(
    text="جواب المصدر",
    citations=(Citation(1, 5, "كتاب", "مؤلف", "3", "body"),),
    deflected=False,
)


class _FakeQA:
    def answer(self, question: str, *, k: int | None = None, filters: object = None) -> Answer:
        return _ANSWER


def test_build_general_qa_service_wires_injected_components() -> None:
    from shamela_rag.factory import build_general_qa_service

    encoder = Bm25Encoder().fit(["نص أول للاختبار", "نص ثانٍ مختلف"])
    service = build_general_qa_service(
        embedder=InMemoryEmbeddingProvider(dims=8),
        reranker=LexicalOverlapReranker(),
        sparse_encoder=encoder,
        translator=InMemoryTranslator(),
        generation_provider=InMemoryGenerationProvider(),
    )
    assert isinstance(service, GeneralQAService)
    assert service._retrieval._config.use_root_expansion is False
    assert service._retrieval._root is None


def test_build_general_qa_service_wires_root_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.factory import build_general_qa_service

    monkeypatch.setenv("SHAMELA_ROOT_EXPANSION_ENABLED", "true")
    get_settings.cache_clear()
    try:
        surface = Bm25Encoder().fit(["نص أول للاختبار", "نص ثانٍ مختلف"])
        root = RootExpansionEncoder(
            load_root_dictionary(Path("tests/fixtures/root_dictionary_sample.jsonl"))
        ).fit(["الصلاة واجبة على كل مسلم"])
        service = build_general_qa_service(
            embedder=InMemoryEmbeddingProvider(dims=8),
            reranker=LexicalOverlapReranker(),
            sparse_encoder=surface,
            translator=InMemoryTranslator(),
            generation_provider=InMemoryGenerationProvider(),
            root_encoder=root,
        )
        assert service._retrieval._config.use_root_expansion is True
        assert service._retrieval._root is not None
    finally:
        get_settings.cache_clear()


def test_build_general_qa_service_requires_root_state_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shamela_rag.factory import build_general_qa_service

    monkeypatch.setenv("SHAMELA_ROOT_EXPANSION_ENABLED", "true")
    monkeypatch.setenv("SHAMELA_ROOT_EXPANSION_STATE_PATH", str(tmp_path / "missing.json"))
    get_settings.cache_clear()
    try:
        surface = Bm25Encoder().fit(["نص أول للاختبار", "نص ثانٍ مختلف"])
        with pytest.raises(FileNotFoundError, match="Root expansion state"):
            build_general_qa_service(
                embedder=InMemoryEmbeddingProvider(dims=8),
                reranker=LexicalOverlapReranker(),
                sparse_encoder=surface,
                translator=InMemoryTranslator(),
                generation_provider=InMemoryGenerationProvider(),
            )
    finally:
        get_settings.cache_clear()


def test_ask_cli_prints_answer_and_citations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shamela_rag.factory as factory

    monkeypatch.setattr(factory, "build_general_qa_service", lambda **_kwargs: _FakeQA())

    assert cli.main(["ask", "سؤال"]) == 0
    out = capsys.readouterr().out
    assert "جواب المصدر" in out
    assert "[1] كتاب" in out


def test_ask_cli_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shamela_rag.factory as factory

    monkeypatch.setattr(factory, "build_general_qa_service", lambda **_kwargs: _FakeQA())

    assert cli.main(["ask", "سؤال", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "جواب المصدر"
    assert payload["citations"][0]["chunk_id"] == 5


def test_create_app_from_settings_injects_service(monkeypatch: pytest.MonkeyPatch) -> None:
    import shamela_rag.factory as factory

    fake = _FakeQA()
    monkeypatch.setattr(factory, "build_general_qa_service", lambda **_kwargs: fake)

    from shamela_rag.api.app import create_app_from_settings

    app = create_app_from_settings()
    assert app.state.qa_service is fake
