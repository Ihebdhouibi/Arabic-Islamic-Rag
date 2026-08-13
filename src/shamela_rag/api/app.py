"""FastAPI application for the general Q&A module (M7-01).

Exposes ``GET /health`` and ``POST /ask`` (Arabic or English question -> cited answer). The
``GeneralQAService`` is injected via ``create_app`` (or ``app.state.qa_service``); when it is not
configured the endpoint returns 503, so the app imports without building heavy models.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from shamela_rag.api.schemas import AnswerResponse, AskRequest, FilterIn
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.filters import RetrievalFilter


def _to_filter(filters: FilterIn | None) -> RetrievalFilter | None:
    if filters is None:
        return None
    return RetrievalFilter(
        book_id=filters.book_id,
        category_id=filters.category_id,
        content_role=filters.content_role,
    )


def _service(request: Request) -> GeneralQAService:
    service: GeneralQAService | None = getattr(request.app.state, "qa_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="QA service not configured")
    return service


def create_app(qa_service: GeneralQAService | None = None) -> FastAPI:
    app = FastAPI(title="Shamela RAG - General QA", version="0.1.0")
    app.state.qa_service = qa_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AnswerResponse)
    def ask(
        payload: AskRequest, service: Annotated[GeneralQAService, Depends(_service)]
    ) -> AnswerResponse:
        try:
            answer = service.answer(
                payload.question, k=payload.k, filters=_to_filter(payload.filters)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AnswerResponse.from_answer(answer)

    return app


app = create_app()


def create_app_from_settings() -> FastAPI:
    """App with the real QA service wired from config (loads heavy backends). For ``uvicorn``."""
    from shamela_rag.factory import build_general_qa_service

    return create_app(build_general_qa_service())
