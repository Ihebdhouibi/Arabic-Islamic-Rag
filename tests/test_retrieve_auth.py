"""Bearer auth on POST /retrieve (issue #157).

The token is a shared secret from settings. When it is unset the check is disabled, which is what
keeps local runs and offline CI working; when it is set, a request without valid credentials is
rejected with 401 and the same error envelope the rest of the endpoint uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from shamela_rag.api.app import create_app
from shamela_rag.config import get_settings
from shamela_rag.retrieval.expand import ExpandedChunkPart, ExpandedPassage
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.service import Deadline, RetrievalConfig, RetrievalOutcome

_TOKEN = "s3cret-integration-token"

_PASSAGE = ExpandedPassage(
    hit_chunk_id=5,
    score=0.95,
    section_id=10,
    chunk_ids=(5,),
    text="قال الشافعي بالقياس",
    parts=(
        ExpandedChunkPart(
            chunk_id=5, source_text="قال الشافعي بالقياس", content_role="body", is_hit=True
        ),
    ),
    payload={
        "book_id": 1,
        "category_id": 16,
        "content_role": "body",
        "book_title": "الرسالة",
        "author": "الشافعي",
        "author_death_hijri": 204,
        "book_type_label": "printed",
        "start_page_id": 42,
        "end_page_id": 42,
        "stable_id": "shamela:1:42:1",
        "retrieval_sources": ["dense"],
    },
)


class _FakeRetrievalService:
    def __init__(self, passages: list[ExpandedPassage]) -> None:
        self._passages = passages
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
        return RetrievalOutcome(
            passages=self._passages,
            candidates_considered=len(self._passages),
            reranked=True,
        )


class _FakeQA:
    def __init__(self, retrieval: _FakeRetrievalService | None) -> None:
        self._retrieval = retrieval


def _client(*, configure: bool = True) -> TestClient:
    retrieval = _FakeRetrievalService([_PASSAGE]) if configure else None
    app = create_app(qa_service=_FakeQA(retrieval) if configure else None)
    if retrieval is not None:
        app.state.retrieval_service = retrieval
    return TestClient(app)


@pytest.fixture
def with_token(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SHAMELA_RETRIEVE_API_TOKEN", _TOKEN)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def without_token(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("SHAMELA_RETRIEVE_API_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _post(client: TestClient, *, authorization: str | None = None):
    headers = {"Authorization": authorization} if authorization is not None else {}
    return client.post(
        "/retrieve",
        json={"query": "سؤال", "deadline_ms": 5000},
        headers=headers,
    )


def test_valid_token_retrieves_normally(with_token: None) -> None:
    resp = _post(_client(), authorization=f"Bearer {_TOKEN}")

    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1


def test_missing_token_is_401(with_token: None) -> None:
    resp = _post(_client())

    assert resp.status_code == 401
    error = resp.json()["error"]
    assert error["code"] == "unauthorized"
    assert error["retryable"] is False


def test_wrong_token_is_401(with_token: None) -> None:
    resp = _post(_client(), authorization="Bearer not-the-token")

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_wrong_scheme_is_401(with_token: None) -> None:
    resp = _post(_client(), authorization=f"Basic {_TOKEN}")

    assert resp.status_code == 401


def test_empty_bearer_value_is_401(with_token: None) -> None:
    resp = _post(_client(), authorization="Bearer    ")

    assert resp.status_code == 401


def test_auth_is_rejected_before_the_service_is_consulted(with_token: None) -> None:
    """An unauthenticated caller must not learn whether a service is configured behind the gate."""
    resp = _post(_client(configure=False))

    assert resp.status_code == 401


def test_no_configured_token_leaves_the_endpoint_open(without_token: None) -> None:
    resp = _post(_client())

    assert resp.status_code == 200


def test_health_is_never_gated(with_token: None) -> None:
    """Probes have no credentials; gating health would make the service look permanently down."""
    resp = _client().get("/health")

    assert resp.status_code in (200, 503)
    assert "error" not in resp.json()
