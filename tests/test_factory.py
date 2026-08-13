from __future__ import annotations

import json

import pytest

from shamela_rag import cli
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
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
