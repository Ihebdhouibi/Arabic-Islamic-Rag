"""Request/response schemas for the general Q&A API (M7-01)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from shamela_rag.generation.answer import Answer


class FilterIn(BaseModel):
    book_id: int | None = None
    category_id: int | None = None
    content_role: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Question in Arabic or English.")
    k: int | None = Field(default=None, ge=1, description="Number of passages to return.")
    filters: FilterIn | None = None


class CitationOut(BaseModel):
    marker: int
    chunk_id: int
    book_title: str
    author: str
    page: str
    content_role: str


class AnswerResponse(BaseModel):
    answer: str
    deflected: bool
    citations: list[CitationOut]

    @classmethod
    def from_answer(cls, answer: Answer) -> AnswerResponse:
        return cls(
            answer=answer.text,
            deflected=answer.deflected,
            citations=[
                CitationOut(
                    marker=citation.marker,
                    chunk_id=citation.chunk_id,
                    book_title=citation.book_title,
                    author=citation.author,
                    page=citation.page,
                    content_role=citation.content_role,
                )
                for citation in answer.citations
            ],
        )
