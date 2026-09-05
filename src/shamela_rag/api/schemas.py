"""Request/response schemas for the general Q&A API (M7-01, M7-retrieve #147)."""

from __future__ import annotations

from typing import Any

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


# ---------------------------------------------------------------------------
# POST /retrieve schemas
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, description="Question in Arabic or English.")
    top_k: int = Field(default=10, ge=1, description="Items after rerank.")
    filters: FilterIn | None = None
    deadline_ms: int = Field(ge=100, description="Hard budget in milliseconds.")


class CategoryOut(BaseModel):
    category_id: int | None = None
    category_name: str = ""
    suggested_domain: str | None = None


class CitationBlock(BaseModel):
    book_id: int
    book_title: str = ""
    author: str = ""
    author_death_hijri: int | None = None
    book_type_label: str = ""
    section_trail: str = ""
    section_confidence: str = ""
    volume: str = ""
    start_page_id: int | None = None
    end_page_id: int | None = None
    printed_page: str | None = None


class RetrieveItem(BaseModel):
    id: str
    text: str
    text_en: str | None = None
    score: float
    rank: int
    content_role: str
    category: CategoryOut
    citation: CitationBlock
    retrieval_sources: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class SearchMeta(BaseModel):
    status: str = "ok"
    candidates_considered: int = 0
    reranked: bool = True
    degraded: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0


class RetrieveResponse(BaseModel):
    items: list[RetrieveItem]
    search: SearchMeta


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# GET /health schemas
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    model: str = ""
    mode: str = ""
    ready: bool = False


class IndexHealth(BaseModel):
    collection: str = ""
    points: int = 0
    ready: bool = False


class HealthResponse(BaseModel):
    status: str
    index: IndexHealth | None = None
    embedder: ComponentHealth | None = None
    reranker: ComponentHealth | None = None
