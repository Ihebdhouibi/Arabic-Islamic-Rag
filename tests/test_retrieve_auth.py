"""Bearer auth on POST /retrieve (issue #157).

The token is a shared secret from settings. When it is unset the check is disabled, which is what
keeps local runs and offline CI working; when it is set, a request without valid credentials is
rejected with 401 and the same error envelope the rest of the endpoint uses.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.test_retrieve_endpoint import _PASSAGE, _client

from shamela_rag.config import get_settings

_TOKEN = "s3cret-integration-token"


@pytest.fixture
def _with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAMELA_RETRIEVE_API_TOKEN", _TOKEN)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def _without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHAMELA_RETRIEVE_API_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _post(client: TestClient, *, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return client.post(
        "/retrieve",
        json={"query": "سؤال", "deadline_ms": 5000},
        headers=headers,
    )


def test_valid_token_retrieves_normally(_with_token: None) -> None:
    client, _ = _client([_PASSAGE])
    resp = _post(client, token=_TOKEN)

    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1


def test_missing_token_is_401(_with_token: None) -> None:
    client, _ = _client([_PASSAGE])
    resp = _post(client)

    assert resp.status_code == 401
    error = resp.json()["error"]
    assert error["code"] == "unauthorized"
    assert error["retryable"] is False


def test_wrong_token_is_401(_with_token: None) -> None:
    client, _ = _client([_PASSAGE])
    resp = _post(client, token="not-the-token")

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_wrong_scheme_is_401(_with_token: None) -> None:
    client, _ = _client([_PASSAGE])
    resp = client.post(
        "/retrieve",
        json={"query": "سؤال", "deadline_ms": 5000},
        headers={"Authorization": f"Basic {_TOKEN}"},
    )

    assert resp.status_code == 401


def test_empty_bearer_value_is_401(_with_token: None) -> None:
    client, _ = _client([_PASSAGE])
    resp = client.post(
        "/retrieve",
        json={"query": "سؤال", "deadline_ms": 5000},
        headers={"Authorization": "Bearer    "},
    )

    assert resp.status_code == 401


def test_auth_is_rejected_before_the_service_is_consulted(_with_token: None) -> None:
    """An unauthenticated caller must not learn whether a service is configured behind the gate."""
    client, _ = _client(configure=False)
    resp = _post(client)

    assert resp.status_code == 401


def test_no_configured_token_leaves_the_endpoint_open(_without_token: None) -> None:
    client, _ = _client([_PASSAGE])
    resp = _post(client)

    assert resp.status_code == 200


def test_health_is_never_gated(_with_token: None) -> None:
    """Probes have no credentials; gating health would make the service look permanently down."""
    client, _ = _client([])
    resp = client.get("/health")

    assert resp.status_code in (200, 503)
    assert "error" not in resp.json()
