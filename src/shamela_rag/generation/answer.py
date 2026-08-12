"""Answer assembly with structured citations (M5-03).

Turns retrieved ``ExpandedPassage`` objects into the general Q&A prompt (doc 07 §6 rules), calls the
generation provider, and returns the answer plus structured citations (book, author, page) each
mapped back to a real chunk id. When evidence is thin (fewer than ``min_passages``), it deflects:
the model is prompted with no sources and no citations are attached, so it cannot fabricate refs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.generation.prompt import PromptPassage, render_general_qa_prompt
from shamela_rag.generation.provider import GenerationProvider
from shamela_rag.retrieval.expand import ExpandedPassage


@dataclass(frozen=True)
class Citation:
    marker: int  # matches the [n] numbering in the prompt/answer
    chunk_id: int
    book_title: str
    author: str
    page: str
    content_role: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    deflected: bool


def _payload_str(passage: ExpandedPassage, key: str) -> str:
    value = passage.payload.get(key)
    return str(value) if value is not None else ""


def _content_role(passage: ExpandedPassage) -> str:
    return _payload_str(passage, "content_role") or ContentRole.BODY.value


def _to_prompt_passage(passage: ExpandedPassage) -> PromptPassage:
    return PromptPassage(
        text=passage.text,
        book_title=_payload_str(passage, "book_title"),
        author=_payload_str(passage, "author"),
        page=_payload_str(passage, "page_id"),
        content_role=_content_role(passage),
    )


def _to_citation(marker: int, passage: ExpandedPassage) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=passage.hit_chunk_id,
        book_title=_payload_str(passage, "book_title"),
        author=_payload_str(passage, "author"),
        page=_payload_str(passage, "page_id"),
        content_role=_content_role(passage),
    )


class AnswerAssembler:
    def __init__(
        self,
        provider: GenerationProvider,
        *,
        max_tokens: int | None = None,
        min_passages: int = 1,
    ) -> None:
        if min_passages < 0:
            raise ValueError(f"min_passages must be >= 0, got {min_passages}")
        self._provider = provider
        self._max_tokens = max_tokens
        self._min_passages = min_passages

    def assemble(self, question: str, passages: Sequence[ExpandedPassage]) -> Answer:
        if len(passages) < self._min_passages:
            prompt = render_general_qa_prompt(question, [])
            text = self._provider.generate(prompt, max_tokens=self._max_tokens)
            return Answer(text=text, citations=(), deflected=True)

        prompt = render_general_qa_prompt(
            question, [_to_prompt_passage(passage) for passage in passages]
        )
        citations = tuple(
            _to_citation(marker, passage) for marker, passage in enumerate(passages, start=1)
        )
        text = self._provider.generate(prompt, max_tokens=self._max_tokens)
        return Answer(text=text, citations=citations, deflected=False)
