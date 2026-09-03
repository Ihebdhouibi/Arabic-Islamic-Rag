"""FastAPI application for the general Q&A module (M7-01, /retrieve M7-02 #147).

Exposes ``GET /health``, ``POST /ask``, and ``POST /retrieve``. The ``GeneralQAService`` is
injected via ``create_app`` (or ``app.state.qa_service``); when it is not configured the endpoint
returns 503, so the app imports without building heavy models.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from shamela_rag.api.schemas import (
    AnswerResponse,
    AskRequest,
    CategoryOut,
    CitationBlock,
    ComponentHealth,
    FilterIn,
    HealthResponse,
    IndexHealth,
    RetrieveItem,
    RetrieveRequest,
    RetrieveResponse,
    SearchMeta,
)
from shamela_rag.chunking.context_header import DEATH_YEAR_UNKNOWN
from shamela_rag.data.domains import suggested_domain_for_category
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.expand import ExpandedPassage, ExpandMode, ExpansionConfig
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.fusion import RETRIEVAL_SOURCES_KEY
from shamela_rag.retrieval.service import RetrievalConfig, RetrievalService


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


def _retrieval_service(request: Request) -> RetrievalService:
    svc: RetrievalService | None = getattr(request.app.state, "retrieval_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="QA service not configured")
    return svc


def _normalize_death_year(value: int | None) -> int | None:
    if value is None or value == DEATH_YEAR_UNKNOWN:
        return None
    return value


def _printed_page(passage: ExpandedPassage) -> str | None:
    start = passage.payload.get("start_page_num")
    end = passage.payload.get("end_page_num")
    if start is None:
        return None
    if end is not None and end != start:
        return f"{start}-{end}"
    return str(start)


def _passage_to_item(passage: ExpandedPassage, rank: int) -> RetrieveItem:
    p = passage.payload
    category_id = p.get("category_id")
    domain = suggested_domain_for_category(category_id)

    hit_text = passage.text
    for part in passage.parts:
        if part.is_hit:
            hit_text = part.source_text
            break

    named_keys = {
        "book_id",
        "category_id",
        "section_id",
        "content_role",
        "page_id",
        "book_title",
        "author",
        "author_death_hijri",
        "book_type_label",
        "part",
        "start_page_id",
        "end_page_id",
        "start_page_num",
        "end_page_num",
        "section_trail",
        "section_confidence",
        "stable_id",
        "context_header",
        RETRIEVAL_SOURCES_KEY,
    }
    raw: dict[str, Any] = {k: v for k, v in p.items() if k not in named_keys}

    return RetrieveItem(
        id=p.get("stable_id", ""),
        text=hit_text,
        text_en=None,
        score=passage.score,
        rank=rank,
        content_role=p.get("content_role", "body"),
        category=CategoryOut(
            category_id=category_id,
            category_name="",
            suggested_domain=domain.value if domain else None,
        ),
        citation=CitationBlock(
            book_id=p.get("book_id", 0),
            book_title=p.get("book_title", ""),
            author=p.get("author", ""),
            author_death_hijri=_normalize_death_year(p.get("author_death_hijri")),
            book_type_label=p.get("book_type_label", ""),
            section_trail=p.get("section_trail") or "",
            section_confidence=p.get("section_confidence") or "",
            volume=p.get("part") or "",
            start_page_id=p.get("start_page_id"),
            end_page_id=p.get("end_page_id"),
            printed_page=_printed_page(passage),
        ),
        retrieval_sources=list(p.get(RETRIEVAL_SOURCES_KEY) or []),
        raw=raw,
    )


def _error_json(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    status: int = 500,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "retryable": retryable}},
    )


def create_app(qa_service: GeneralQAService | None = None) -> FastAPI:
    app = FastAPI(title="Shamela RAG - General QA", version="0.1.0")
    app.state.qa_service = qa_service
    app.state.retrieval_service = getattr(qa_service, "_retrieval", None)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc: GeneralQAService | None = getattr(app.state, "qa_service", None)
        if svc is None:
            return HealthResponse(status="unavailable")
        rs: RetrievalService | None = getattr(app.state, "retrieval_service", None)
        index_health: IndexHealth | None = None
        embedder_health: ComponentHealth | None = None
        reranker_health: ComponentHealth | None = None
        if rs is not None:
            from shamela_rag.config import get_settings

            settings = get_settings()
            try:
                info = rs._dense._store.client.get_collection(settings.qdrant_collection)
                index_health = IndexHealth(
                    collection=settings.qdrant_collection,
                    points=info.points_count or 0,
                    ready=info.status.name == "GREEN",
                )
            except Exception:
                index_health = IndexHealth(collection=settings.qdrant_collection, ready=False)

            embedder_health = ComponentHealth(
                model=settings.dense_embedding_model,
                mode=settings.embedding_backend,
                ready=True,
            )
            reranker_health = ComponentHealth(
                model=settings.reranker_model,
                mode="cross-encoder",
                ready=True,
            )
        return HealthResponse(
            status="ok",
            index=index_health,
            embedder=embedder_health,
            reranker=reranker_health,
        )

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

    @app.post("/retrieve", response_model=None)
    def retrieve(
        payload: RetrieveRequest,
        svc: Annotated[RetrievalService, Depends(_retrieval_service)],
    ) -> RetrieveResponse | JSONResponse:
        start_ms = time.monotonic_ns() // 1_000_000
        degraded: list[str] = []

        config = RetrievalConfig(
            final_k=payload.top_k,
            rerank_top_k=payload.top_k,
            expansion=ExpansionConfig(mode=ExpandMode.NONE),
        )

        try:
            passages = svc.retrieve(
                payload.query,
                k=payload.top_k,
                filters=_to_filter(payload.filters),
                config=config,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            return _error_json(
                "retrieval_error",
                str(exc),
                retryable=True,
                status=502,
            )

        elapsed = time.monotonic_ns() // 1_000_000 - start_ms
        if elapsed > payload.deadline_ms:
            degraded.append("deadline_hit")

        items = [_passage_to_item(p, rank) for rank, p in enumerate(passages, 1)]
        status = "partial" if degraded else "ok"

        return RetrieveResponse(
            items=items,
            search=SearchMeta(
                status=status,
                candidates_considered=len(passages),
                reranked=True,
                degraded=degraded,
                elapsed_ms=elapsed,
            ),
        )

    return app


app = create_app()


def create_app_from_settings() -> FastAPI:
    """App with the real QA service wired from config (loads heavy backends). For ``uvicorn``."""
    from shamela_rag.factory import build_general_qa_service

    return create_app(build_general_qa_service())
