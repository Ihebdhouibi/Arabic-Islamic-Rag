from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shamela_rag.api.app import create_app
from shamela_rag.generation.answer import Answer, Citation
from shamela_rag.retrieval.filters import RetrievalFilter


class _FakeQA:
    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.last_filters: RetrievalFilter | None = None
        self.last_k: int | None = None

    def answer(
        self, question: str, *, k: int | None = None, filters: RetrievalFilter | None = None
    ) -> Answer:
        self.last_k = k
        self.last_filters = filters
        return self._answer


_ANSWER = Answer(
    text="قال الشافعي بالقياس",
    citations=(
        Citation(
            1,
            "shamela:1:42:1",
            5,
            "الرسالة",
            "الشافعي",
            "42",
            "body",
            category=16,
            snippet="قال الشافعي",
        ),
    ),
    deflected=False,
)


def _client(qa: object | None = None) -> TestClient:
    app = create_app()
    if qa is not None:
        app.state.qa_service = qa
    return TestClient(app)


def test_health_ok() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_cited_answer() -> None:
    fake = _FakeQA(_ANSWER)
    response = _client(fake).post("/ask", json={"question": "ما رأي الشافعي؟", "k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "قال الشافعي بالقياس"
    assert body["deflected"] is False
    assert body["citations"] == [
        {
            "marker": 1,
            "id": "shamela:1:42:1",
            "chunk_id": 5,
            "book_title": "الرسالة",
            "author": "الشافعي",
            "page": "42",
            "category": 16,
            "content_role": "body",
            "is_footnote": False,
            "snippet": "قال الشافعي",
        }
    ]
    assert fake.last_k == 3


def test_ask_passes_filters_through() -> None:
    fake = _FakeQA(_ANSWER)
    _client(fake).post(
        "/ask",
        json={"question": "سؤال", "filters": {"book_id": 7, "content_role": "body"}},
    )
    assert fake.last_filters == RetrievalFilter(book_id=7, content_role="body")


def test_ask_footnote_citation_is_marked() -> None:
    footnote = Answer(
        text="جواب",
        citations=(
            Citation(1, "shamela:1:3:1", 9, "كتاب", "مؤلف", "3", "footnote", snippet="حاشية"),
        ),
        deflected=False,
    )
    body = _client(_FakeQA(footnote)).post("/ask", json={"question": "س"}).json()
    citation = body["citations"][0]
    assert citation["is_footnote"] is True
    assert citation["content_role"] == "footnote"
    assert citation["snippet"] == "حاشية"


def test_ask_empty_question_is_422() -> None:
    response = _client(_FakeQA(_ANSWER)).post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_without_configured_service_is_503() -> None:
    response = _client().post("/ask", json={"question": "سؤال"})
    assert response.status_code == 503


@pytest.mark.parametrize("bad_k", [0, -1])
def test_ask_rejects_non_positive_k(bad_k: int) -> None:
    response = _client(_FakeQA(_ANSWER)).post("/ask", json={"question": "س", "k": bad_k})
    assert response.status_code == 422
