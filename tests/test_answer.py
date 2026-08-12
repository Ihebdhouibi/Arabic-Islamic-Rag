from __future__ import annotations

from typing import Any

import pytest

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.generation.answer import Answer, AnswerAssembler, Citation
from shamela_rag.generation.provider import InMemoryGenerationProvider
from shamela_rag.retrieval.expand import ExpandedPassage


def _passage(
    chunk_id: int,
    *,
    text: str,
    book_title: str = "الكتاب",
    author: str = "المؤلف",
    page_id: int = 7,
    content_role: str = ContentRole.BODY.value,
) -> ExpandedPassage:
    payload: dict[str, Any] = {
        "book_title": book_title,
        "author": author,
        "page_id": page_id,
        "content_role": content_role,
        "book_id": 1,
    }
    return ExpandedPassage(
        hit_chunk_id=chunk_id,
        score=1.0,
        section_id=1,
        chunk_ids=(chunk_id,),
        text=text,
        parts=(),
        payload=payload,
    )


def test_assemble_returns_answer_and_citations() -> None:
    provider = InMemoryGenerationProvider()
    assembler = AnswerAssembler(provider)
    passages = [
        _passage(11, text="قال الشافعي بالقياس"),
        _passage(22, text="حاشية المحقق", content_role=ContentRole.FOOTNOTE.value),
    ]

    answer = assembler.assemble("ما رأي الشافعي؟", passages)

    assert isinstance(answer, Answer)
    assert answer.deflected is False
    assert answer.text != ""
    assert [c.chunk_id for c in answer.citations] == [11, 22]
    assert [c.marker for c in answer.citations] == [1, 2]
    first = answer.citations[0]
    assert (first.book_title, first.author, first.page) == ("الكتاب", "المؤلف", "7")
    assert answer.citations[1].content_role == ContentRole.FOOTNOTE.value


def test_prompt_receives_passage_metadata() -> None:
    provider = InMemoryGenerationProvider()
    AnswerAssembler(provider).assemble("سؤال", [_passage(1, text="نص المصدر")])

    prompt = provider.calls[0].prompt
    assert "نص المصدر" in prompt
    assert "book=الكتاب" in prompt
    assert "Cite every claim" in prompt


def test_thin_evidence_deflects_without_citations() -> None:
    provider = InMemoryGenerationProvider()
    answer = AnswerAssembler(provider).assemble("سؤال بلا مصادر", [])

    assert answer.deflected is True
    assert answer.citations == ()
    assert "no sources retrieved" in provider.calls[0].prompt


def test_min_passages_threshold_deflects() -> None:
    provider = InMemoryGenerationProvider()
    assembler = AnswerAssembler(provider, min_passages=2)

    answer = assembler.assemble("سؤال", [_passage(1, text="مصدر واحد")])

    assert answer.deflected is True
    assert answer.citations == ()


def test_every_citation_maps_to_a_passage_chunk_id() -> None:
    passages = [_passage(101, text="alpha"), _passage(202, text="beta"), _passage(303, text="g")]
    answer = AnswerAssembler(InMemoryGenerationProvider()).assemble("q", passages)

    passage_ids = {p.hit_chunk_id for p in passages}
    assert all(isinstance(c, Citation) for c in answer.citations)
    assert all(c.chunk_id in passage_ids for c in answer.citations)


def test_negative_min_passages_rejected() -> None:
    with pytest.raises(ValueError, match="min_passages"):
        AnswerAssembler(InMemoryGenerationProvider(), min_passages=-1)
