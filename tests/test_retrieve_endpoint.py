"""Tests for POST /retrieve and enhanced GET /health (issue #147)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from shamela_rag.api.app import create_app
from shamela_rag.generation.answer import Answer
from shamela_rag.retrieval.expand import ExpandedChunkPart, ExpandedPassage
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.service import Deadline, RetrievalConfig, RetrievalOutcome


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
        "category_name": "الفقه الشافعي",
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
    def __init__(
        self,
        passages: list[ExpandedPassage],
        *,
        outcome: RetrievalOutcome | None = None,
    ) -> None:
        self._passages = passages
        self._outcome = outcome
        self.last_filters: RetrievalFilter | None = None
        self.last_k: int | None = None
        self.last_deadline: Deadline | None = None
        self._dense = MagicMock()
        self._dense._store = MagicMock()
        self._reranker = MagicMock()

    def retrieve_with_outcome(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilter | None = None,
        config: RetrievalConfig | None = None,
        deadline: Deadline | None = None,
    ) -> RetrievalOutcome:
        if not question.strip():
            raise ValueError("question must be non-empty")
        self.last_filters = filters
        self.last_k = k
        self.last_deadline = deadline
        if self._outcome is not None:
            return self._outcome
        return RetrievalOutcome(
            passages=self._passages,
            candidates_considered=len(self._passages),
            reranked=True,
        )

    def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        filters: RetrievalFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[ExpandedPassage]:
        return self.retrieve_with_outcome(question, k=k, filters=filters, config=config).passages


class _FakeQA:
    def __init__(self, answer: Answer, retrieval: _FakeRetrievalService | None = None) -> None:
        self._answer = answer
        self._retrieval = retrieval or _FakeRetrievalService([])

    def answer(
        self, question: str, *, k: int | None = None, filters: RetrievalFilter | None = None
    ) -> Answer:
        return self._answer


def _client(
    passages: list[ExpandedPassage] | None = None,
    *,
    configure: bool = True,
    outcome: RetrievalOutcome | None = None,
) -> tuple[TestClient, _FakeRetrievalService | None]:
    fake_retrieval = _FakeRetrievalService(passages or [], outcome=outcome) if configure else None
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

    fake.retrieve_with_outcome = _boom
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


def _ready_index(client_mock: MagicMock) -> None:
    """Make the fake Qdrant client report a live, populated collection."""
    info = MagicMock()
    info.points_count = 1234
    info.status.name = "GREEN"
    client_mock.get_collection.return_value = info


def test_health_ok_when_everything_is_ready() -> None:
    client, fake = _client([])
    assert fake is not None
    _ready_index(fake._dense._store.client)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index"]["ready"] is True
    assert body["index"]["points"] == 1234
    assert body["embedder"]["ready"] is True
    assert body["reranker"]["ready"] is True


def test_health_is_503_when_index_is_not_reachable() -> None:
    client, fake = _client([])
    assert fake is not None
    fake._dense._store.client.get_collection.side_effect = ConnectionError("qdrant down")

    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["index"]["ready"] is False


def test_health_is_503_when_collection_is_empty() -> None:
    client, fake = _client([])
    assert fake is not None
    info = MagicMock()
    info.points_count = 0
    info.status.name = "GREEN"
    fake._dense._store.client.get_collection.return_value = info

    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["index"]["ready"] is False


def test_health_unavailable_without_service_is_503() -> None:
    client, _ = _client(configure=False)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"


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


def test_retrieve_reports_partial_when_stages_were_skipped() -> None:
    """The endpoint must surface what the pipeline skipped, not claim a clean run."""
    degraded_outcome = RetrievalOutcome(
        passages=[_PASSAGE],
        candidates_considered=37,
        reranked=False,
        degraded=("sparse_skipped", "rerank_skipped", "deadline_hit"),
        elapsed_ms=8123,
    )
    client, _ = _client([_PASSAGE], outcome=degraded_outcome)
    resp = client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 5000})

    assert resp.status_code == 200
    search = resp.json()["search"]
    assert search["status"] == "partial"
    assert search["reranked"] is False
    assert search["degraded"] == ["sparse_skipped", "rerank_skipped", "deadline_hit"]
    assert search["candidates_considered"] == 37
    assert search["elapsed_ms"] == 8123
    # Fails open: partial still returns the evidence it managed to gather.
    assert len(resp.json()["items"]) == 1


def test_retrieve_passes_the_deadline_budget_to_the_pipeline() -> None:
    client, fake = _client([_PASSAGE])
    client.post("/retrieve", json={"query": "سؤال", "deadline_ms": 2500})

    assert fake is not None
    assert fake.last_deadline is not None
    assert fake.last_deadline._budget_ms == 2500


def test_retrieve_candidates_considered_is_the_pool_not_the_page() -> None:
    outcome = RetrievalOutcome(passages=[_PASSAGE], candidates_considered=50, reranked=True)
    client, _ = _client([_PASSAGE], outcome=outcome)
    body = client.post("/retrieve", json={"query": "س", "deadline_ms": 5000}).json()

    assert len(body["items"]) == 1
    assert body["search"]["candidates_considered"] == 50


def test_retrieve_exposes_category_name() -> None:
    client, _ = _client([_PASSAGE])
    body = client.post("/retrieve", json={"query": "س", "deadline_ms": 5000}).json()
    category = body["items"][0]["category"]

    assert category["category_name"] == "الفقه الشافعي"
    assert category["category_id"] == 16
    assert category["suggested_domain"] == "fiqh"
    # Named fields must not also leak into the passthrough bucket.
    assert "category_name" not in body["items"][0]["raw"]


def test_retrieve_missing_stable_id_is_a_loud_non_retryable_error() -> None:
    broken = ExpandedPassage(
        hit_chunk_id=99,
        score=0.5,
        section_id=None,
        chunk_ids=(99,),
        text="نص",
        parts=(ExpandedChunkPart(chunk_id=99, source_text="نص", content_role="body", is_hit=True),),
        payload={"book_id": 1, "category_id": 1, "content_role": "body"},  # no stable_id
    )
    client, _ = _client([broken])
    resp = client.post("/retrieve", json={"query": "س", "deadline_ms": 5000})

    assert resp.status_code == 500
    error = resp.json()["error"]
    assert error["code"] == "missing_stable_id"
    assert error["retryable"] is False
