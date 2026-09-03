"""Tests for POST /retrieve and enhanced GET /health (issue #147)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from shamela_rag.api.app import create_app
from shamela_rag.generation.answer import Answer
from shamela_rag.retrieval.expand import ExpandedChunkPart, ExpandedPassage
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.service import RetrievalConfig


class _StubReranker:
    contributes_rerank_source = False

    def rerank(self, query, candidates, top_k=10):
        from shamela_rag.retrieval.rerank import RerankedChunk

        return [
            RerankedChunk(chunk_id=c.chunk_id, score=1.0 - i * 0.1, text=c.text, payload=c.payload)
            for i, c in enumerate(candidates[:top_k])
        ]


class _StubGenerator:
    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return "stub"


_PASSAGE = ExpandedPassage(
    hit_chunk_id=5,
    score=0.95,
    section_id=10,
    chunk_ids=(5,),
    text="header\n\nقال الشافعي بالقياس",
    parts=(
        ExpandedChunkPart(
            chunk_id=5, source_text="قال الشافعي بالقياس", content_role="body", is_hit=True
        ),
    ),
    payload={
        "book_id": 1,
        "category_id": 16,
        "section_id": 10,
        "content_role": "body",
        "page_id": 42,
        "book_title": "الرسالة",
        "author": "الشافعي",
        "author_death_hijri": 204,
        "book_type_label": "printed",
        "part": "1",
        "start_page_id": 42,
        "end_page_id": 42,
        "start_page_num": 100,
        "end_page_num": 100,
        "section_trail": "كتاب القياس > فصل في البيان",
        "section_confidence": "high",
        "stable_id": "shamela:1:42:1",
        "retrieval_sources": ["dense", "bm25", "rerank"],
    },
)

_PASSAGE_UNKNOWN_DEATH = ExpandedPassage(
    hit_chunk_id=6,
    score=0.8,
    section_id=11,
    chunk_ids=(6,),
    text="نص مجهول المؤلف",
    parts=(
        ExpandedChunkPart(
            chunk_id=6, source_text="نص مجهول المؤلف", content_role="body", is_hit=True
        ),
    ),
    payload={
        "book_id": 2,
        "category_id": 3,
        "section_id": 11,
        "content_role": "body",
        "page_id": 1,
        "book_title": "كتاب",
        "author": "مجهول",
        "author_death_hijri": 99999,
        "book_type_label": "manuscript",
        "part": None,
        "start_page_id": 1,
        "end_page_id": 1,
        "start_page_num": None,
        "end_page_num": None,
        "section_trail": None,
        "section_confidence": None,
        "stable_id": "shamela:2:1:1",
        "retrieval_sources": [],
    },
)


class _FakeRetrievalService:
    def __init__(self, passages: list[ExpandedPassage]) -> None:
        self._passages = passages
        self.last_filters: RetrievalFilter | None = None
        self.last_k: int | None = None
        self._dense = MagicMock()
        self._dense._store = MagicMock()

    def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[ExpandedPassage]:
        if not question.strip():
            raise ValueError("question must be non-empty")
        self.last_filters = filters
        self.last_k = k
        return self._passages


class _FakeQA:
    def __init__(self, answer: Answer, retrieval: _FakeRetrievalService | None = None) -> None:
        self._answer = answer
        self._retrieval = retrieval or _FakeRetrievalService([])

    def answer(
        self, question: str, *, k: int | None = None, filters: RetrievalFilter | None = None
    ) -> Answer:
        return self._answer


def _client(
    passages: list[ExpandedPassage] | None = None, *, configure: bool = True
) -> tuple[TestClient, _FakeRetrievalService | None]:
    fake_retrieval = _FakeRetrievalService(passages or []) if configure else None
    fake_answer = Answer(text="stub", citations=(), deflected=True)
    fake_qa = _FakeQA(fake_answer, fake_retrieval) if configure else None

    app = create_app(qa_service=fake_qa)
    if fake_retrieval is not None:
        app.state.retrieval_service = fake_retrieval
    return TestClient(app), fake_retrieval


def test_retrieve_returns_evidence_bundle() -> None:
    client, _ = _client([_PASSAGE])
    resp = client.post(
        "/retrieve", json={"query": "ما رأي الشافعي في القياس؟", "deadline_ms": 5000}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["search"]["status"] == "ok"
    assert len(body["items"]) == 1

    item = body["items"][0]
    assert item["id"] == "shamela:1:42:1"
    assert item["text"] == "قال الشافعي بالقياس"
    assert item["text_en"] is None
    assert item["score"] == 0.95
    assert item["rank"] == 1
    assert item["content_role"] == "body"

    cat = item["category"]
    assert cat["category_id"] == 16
    assert cat["suggested_domain"] == "fiqh"

    cit = item["citation"]
    assert cit["book_id"] == 1
    assert cit["book_title"] == "الرسالة"
    assert cit["author"] == "الشافعي"
    assert cit["author_death_hijri"] == 204
    assert cit["book_type_label"] == "printed"
    assert cit["section_trail"] == "كتاب القياس > فصل في البيان"
    assert cit["section_confidence"] == "high"
    assert cit["volume"] == "1"
    assert cit["start_page_id"] == 42
    assert cit["printed_page"] == "100"

    assert item["retrieval_sources"] == ["dense", "bm25", "rerank"]


def test_retrieve_death_year_sentinel_becomes_null() -> None:
    client, _ = _client([_PASSAGE_UNKNOWN_DEATH])
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["citation"]["author_death_hijri"] is None


def test_retrieve_no_printed_page_is_null() -> None:
    client, _ = _client([_PASSAGE_UNKNOWN_DEATH])
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})
    item = resp.json()["items"][0]
    assert item["citation"]["printed_page"] is None
    assert item["citation"]["volume"] == ""


def test_retrieve_empty_results_is_200() -> None:
    client, _ = _client([])
    resp = client.post("/retrieve", json={"query": "nonexistent", "deadline_ms": 5000})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["search"]["status"] == "ok"


def test_retrieve_empty_query_is_400() -> None:
    client, _ = _client([])
    resp = client.post("/retrieve", json={"query": "   ", "deadline_ms": 5000})
    assert resp.status_code == 400


def test_retrieve_missing_deadline_is_422() -> None:
    client, _ = _client([])
    resp = client.post("/retrieve", json={"query": "سؤال"})
    assert resp.status_code == 422


def test_retrieve_without_service_is_503() -> None:
    client, _ = _client(configure=False)
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})
    assert resp.status_code == 503


def test_retrieve_passes_filters() -> None:
    client, fake = _client([_PASSAGE])
    client.post(
        "/retrieve",
        json={
            "query": "سؤال",
            "deadline_ms": 5000,
            "filters": {"book_id": 7, "content_role": "body"},
        },
    )
    assert fake is not None
    assert fake.last_filters == RetrievalFilter(book_id=7, content_role="body")


def test_retrieve_backend_error_returns_502() -> None:
    client, fake = _client([])
    assert fake is not None

    def _boom(*a, **kw):
        raise ConnectionError("Qdrant down")

    fake.retrieve = _boom
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})
    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["code"] == "retrieval_error"
    assert body["error"]["retryable"] is True


def test_retrieve_bad_top_k_is_422() -> None:
    client, _ = _client([])
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000, "top_k": 0})
    assert resp.status_code == 422


def test_retrieve_domain_null_for_unmapped_category() -> None:
    client, _ = _client([_PASSAGE_UNKNOWN_DEATH])
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})
    item = resp.json()["items"][0]
    assert item["category"]["suggested_domain"] is None


def test_retrieve_printed_page_range() -> None:
    p = ExpandedPassage(
        hit_chunk_id=7,
        score=0.5,
        section_id=1,
        chunk_ids=(7,),
        text="text",
        parts=(
            ExpandedChunkPart(chunk_id=7, source_text="text", content_role="body", is_hit=True),
        ),
        payload={
            "book_id": 1,
            "category_id": 1,
            "content_role": "body",
            "stable_id": "shamela:1:1:1",
            "book_title": "",
            "author": "",
            "author_death_hijri": None,
            "book_type_label": "",
            "start_page_id": 1,
            "end_page_id": 2,
            "start_page_num": 50,
            "end_page_num": 51,
            "retrieval_sources": [],
        },
    )
    client, _ = _client([p])
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})
    assert resp.json()["items"][0]["citation"]["printed_page"] == "50-51"


def test_health_ok_with_service() -> None:
    client, _ = _client([])
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_health_unavailable_without_service() -> None:
    client, _ = _client(configure=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unavailable"


def test_retrieve_text_is_hit_source_not_expanded() -> None:
    multi_part = ExpandedPassage(
        hit_chunk_id=5,
        score=0.9,
        section_id=10,
        chunk_ids=(4, 5, 6),
        text="context header\n\nprev\n\nhit text\n\nnext",
        parts=(
            ExpandedChunkPart(chunk_id=4, source_text="prev", content_role="body", is_hit=False),
            ExpandedChunkPart(chunk_id=5, source_text="hit text", content_role="body", is_hit=True),
            ExpandedChunkPart(chunk_id=6, source_text="next", content_role="body", is_hit=False),
        ),
        payload={
            "book_id": 1,
            "category_id": 1,
            "content_role": "body",
            "stable_id": "shamela:1:1:1",
            "book_title": "",
            "author": "",
            "author_death_hijri": None,
            "book_type_label": "",
            "start_page_id": 1,
            "end_page_id": 1,
            "start_page_num": None,
            "end_page_num": None,
            "retrieval_sources": [],
        },
    )
    client, _ = _client([multi_part])
    resp = client.post("/retrieve", json={"query": "q", "deadline_ms": 5000})
    assert resp.json()["items"][0]["text"] == "hit text"
