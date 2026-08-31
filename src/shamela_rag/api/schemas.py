"""Request/response schemas for the general Q&A API (M7-01)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from shamela_rag.chunking.content_roles import ContentRole
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
    marker: int = Field(description="1-based source number matching the answer text.")
    id: str = Field(description="Stable citation id (survives re-ingest).")
    chunk_id: int = Field(description="Postgres serial id; kept during transition.")
    book_title: str
    author: str
    page: str
    category: int | None = Field(default=None, description="Corpus category id of the book.")
    content_role: str = Field(description="body or footnote.")
    is_footnote: bool = Field(description="True for editor/muhaqqiq notes, not the author's words.")
    snippet: str = Field(description="Short excerpt of the cited passage.")


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
                    id=citation.id,
                    chunk_id=citation.chunk_id,
                    book_title=citation.book_title,
                    author=citation.author,
                    page=citation.page,
                    category=citation.category,
                    content_role=citation.content_role,
                    is_footnote=citation.content_role == ContentRole.FOOTNOTE.value,
                    snippet=citation.snippet,
                )
                for citation in answer.citations
            ],
        )
